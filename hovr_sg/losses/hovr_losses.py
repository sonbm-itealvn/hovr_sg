from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sigmoid_focal(logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def make_multihot(indices: Sequence[Sequence[int]], num_classes: int, device: torch.device) -> Tensor:
    out = torch.zeros(len(indices), num_classes, device=device)
    for row, values in enumerate(indices):
        for index in values:
            if 0 <= index < num_classes:
                out[row, index] = 1.0
    return out


def prototype_contrastive(z: Tensor, prototypes: Tensor, target: Tensor, temperature: float = 0.07) -> Tensor:
    logits = torch.einsum("nd,cd->nc", F.normalize(z.float(), p=2.0, dim=-1, eps=1e-5).to(z.dtype), F.normalize(prototypes.float(), p=2.0, dim=-1, eps=1e-5).to(prototypes.dtype)) / temperature
    return F.cross_entropy(logits, target)


def ancestor_consistency(
    leaf_logits: Tensor,
    group_logits: Tensor,
    leaf_to_groups: Dict[int, List[int]],
) -> Tensor:
    """Penalize a leaf being more confident than one of its ancestors."""
    if not leaf_to_groups:
        return leaf_logits.sum() * 0.0
    losses = []
    leaf_prob = leaf_logits.sigmoid()
    group_prob = group_logits.sigmoid()
    for leaf_idx, group_indices in leaf_to_groups.items():
        for group_idx in group_indices:
            losses.append(F.relu(leaf_prob[..., leaf_idx] - group_prob[..., group_idx]).mean())
    return torch.stack(losses).mean() if losses else leaf_logits.sum() * 0.0


def sibling_margin(
    z_leaf: Tensor,
    positive_labels: Tensor,
    sibling_map: Dict[int, List[int]],
    prototypes: Tensor,
    margin: float = 0.20,
) -> Tensor:
    """Keep positive prototype above sibling hard negatives in cosine space."""
    if z_leaf.numel() == 0:
        return z_leaf.sum() * 0.0
    z = F.normalize(z_leaf.float(), p=2.0, dim=-1, eps=1e-5).to(z_leaf.dtype)
    p = F.normalize(prototypes.float(), p=2.0, dim=-1, eps=1e-5).to(prototypes.dtype)
    losses = []
    flat_z, flat_y = z.reshape(-1, z.shape[-1]), positive_labels.reshape(-1)
    for vector, label in zip(flat_z, flat_y.tolist()):
        if label not in sibling_map:
            continue
        pos = vector @ p[label]
        for sibling in sibling_map[label]:
            if 0 <= sibling < p.shape[0]:
                neg = vector @ p[sibling]
                losses.append(F.relu(margin - pos + neg))
    return torch.stack(losses).mean() if losses else z_leaf.sum() * 0.0


def relationness_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return sigmoid_focal(logits, targets.float())


def relation_prototype_loss(
    z_rel: Tensor, relation_prototypes: Tensor, targets: Tensor, ignore_index: int = -1
) -> Tensor:
    valid = targets != ignore_index
    if not valid.any():
        return z_rel.sum() * 0.0
    return prototype_contrastive(z_rel[valid], relation_prototypes, targets[valid])


class UncertaintyWeighting(nn.Module):
    """Multi-task loss balancing via learned homoscedastic uncertainty.

    Implements the approach of Kendall, Gal & Cipolla (CVPR 2018).
    Each loss term *k* has a learnable log-variance ``log_sigma_k``;
    the effective weight becomes ``1 / (2 * sigma_k^2)`` with an
    additive ``log(sigma_k)`` regulariser that prevents all variances
    from growing unboundedly.

    When this module is disabled (not instantiated), the training
    script falls back to the fixed-weight scheme.
    """

    def __init__(self, loss_names: Sequence[str]):
        super().__init__()
        # Initialise log(sigma^2) = 0 → sigma = 1 → weight = 0.5
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in loss_names
        })

    def forward(
        self,
        losses: Dict[str, Tensor],
        fixed_weights: Dict[str, float],
    ) -> Tensor:
        """Compute uncertainty-weighted total loss.

        Fixed weights are still multiplied so that the user's relative
        priorities are respected, but the magnitude is auto-balanced.
        """
        total = losses[next(iter(losses))].sum() * 0.0  # zero on correct device
        for name, loss_val in losses.items():
            if name == "total":
                continue
            log_var = self.log_vars.get(name)
            if log_var is None:
                total = total + fixed_weights.get(name, 1.0) * loss_val
                continue
            precision = torch.exp(-log_var[0])  # 1 / sigma^2
            total = total + fixed_weights.get(name, 1.0) * (0.5 * precision * loss_val + 0.5 * log_var[0])
        return total


def build_loss(
    outputs: Dict[str, Tensor], targets: Dict[str, Tensor],
    leaf_to_groups: Dict[int, List[int]], sibling_map: Dict[int, List[int]],
    leaf_prototypes: Tensor, relation_prototypes: Tensor,
    weights: Dict[str, float],
) -> Dict[str, Tensor]:
    losses: Dict[str, Tensor] = {}
    losses["objectness"] = sigmoid_focal(outputs["objectness_logits"], targets["objectness"])
    losses["leaf"] = prototype_contrastive(outputs["z_leaf"], leaf_prototypes, targets["leaf_index"])
    losses["group"] = sigmoid_focal(outputs["group_logits"], targets["group_multihot"])
    losses["ancestor"] = ancestor_consistency(outputs["leaf_logits"], outputs["group_logits"], leaf_to_groups)
    losses["sibling"] = sibling_margin(outputs["z_leaf"], targets["leaf_index"], sibling_map, leaf_prototypes)
    losses["box_l1"] = F.l1_loss(outputs["boxes"], targets["boxes"])
    losses["relationness"] = relationness_loss(outputs["relationness_logits"], targets["relationness"])
    relation_outputs = outputs.get("relations", {})
    if "relation_logits" in relation_outputs and "predicate_index" in targets:
        losses["predicate"] = relation_prototype_loss(
            relation_outputs["z_rel"], relation_prototypes, targets["predicate_index"]
        )
    else:
        losses["predicate"] = outputs["z_leaf"].sum() * 0.0
    total = outputs["z_leaf"].sum() * 0.0
    for name, value in losses.items():
        total = total + weights.get(name, 1.0) * value
    losses["total"] = total
    return losses
