from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(max(depth - 1, 1)):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HierarchicalPrototypeHead(nn.Module):
    def __init__(self, d_model: int, d_latent: int = 512):
        super().__init__()
        self.d_latent = int(d_latent)
        self.leaf_proj = nn.Linear(d_model, self.d_latent)
        self.group_proj = nn.Linear(d_model, self.d_latent)
        self.log_tau_leaf = nn.Parameter(torch.log(torch.tensor(0.07)))
        self.log_tau_group = nn.Parameter(torch.log(torch.tensor(0.07)))

    def forward(self, slots: Tensor, leaf_text: Tensor, group_text: Tensor) -> Dict[str, Tensor]:
        z_leaf = F.normalize(self.leaf_proj(slots), dim=-1)
        z_group = F.normalize(self.group_proj(slots), dim=-1)
        leaf_text = F.normalize(leaf_text, dim=-1)
        group_text = F.normalize(group_text, dim=-1)
        leaf_logits = torch.einsum("bqd,cd->bqc", z_leaf, leaf_text)
        group_logits = torch.einsum("bqd,gd->bqg", z_group, group_text)
        return {
            "z_leaf": z_leaf,
            "z_group": z_group,
            "leaf_logits": leaf_logits / self.log_tau_leaf.exp().clamp_min(1e-4),
            "group_logits": group_logits / self.log_tau_group.exp().clamp_min(1e-4),
        }


class SparseRelationDecoder(nn.Module):
    def __init__(self, d_model: int, d_latent: int = 512, hidden: int = 512):
        super().__init__()
        self.pair_mlp = MLP(4 * d_model + 9, hidden, hidden, depth=3)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=4 * hidden,
            batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=2)
        self.relationness = nn.Linear(hidden, 1)
        self.rel_proj = nn.Linear(hidden, d_latent)
        self.log_tau_rel = nn.Parameter(torch.log(torch.tensor(0.07)))

    @staticmethod
    def geometry(boxes: Tensor, s: Tensor, o: Tensor) -> Tensor:
        bs, bo = boxes[:, s], boxes[:, o]
        cs, co = (bs[..., :2] + bs[..., 2:]) * 0.5, (bo[..., :2] + bo[..., 2:]) * 0.5
        ws = (bs[..., 2] - bs[..., 0]).clamp_min(1e-6)
        hs = (bs[..., 3] - bs[..., 1]).clamp_min(1e-6)
        wo = (bo[..., 2] - bo[..., 0]).clamp_min(1e-6)
        ho = (bo[..., 3] - bo[..., 1]).clamp_min(1e-6)
        ix1, iy1 = torch.maximum(bs[..., 0], bo[..., 0]), torch.maximum(bs[..., 1], bo[..., 1])
        ix2, iy2 = torch.minimum(bs[..., 2], bo[..., 2]), torch.minimum(bs[..., 3], bo[..., 3])
        inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
        iou = inter / (ws * hs + wo * ho - inter + 1e-6)
        return torch.stack([
            (co[..., 0] - cs[..., 0]) / ws, (co[..., 1] - cs[..., 1]) / hs,
            torch.log(wo / ws), torch.log(ho / hs), torch.log(ws * hs), torch.log(wo * ho),
            iou, (co[..., 0] - cs[..., 0]) / wo, (co[..., 1] - cs[..., 1]) / ho,
        ], dim=-1)

    def forward(
        self, slots: Tensor, boxes: Tensor, object_scores: Tensor,
        relation_text: Optional[Tensor] = None, top_m: int = 100, top_k_pairs: int = 256,
    ) -> Dict[str, Tensor]:
        bsz, _, dim = slots.shape
        m = min(top_m, slots.shape[1])
        chosen = object_scores.topk(m, dim=1).indices
        selected = torch.gather(slots, 1, chosen[..., None].expand(-1, -1, dim))
        selected_boxes = torch.gather(boxes, 1, chosen[..., None].expand(-1, -1, 4))
        idx_s, idx_o = torch.meshgrid(
            torch.arange(m, device=slots.device), torch.arange(m, device=slots.device), indexing="ij"
        )
        keep = idx_s != idx_o
        idx_s, idx_o = idx_s[keep], idx_o[keep]
        s_feat, o_feat = selected[:, idx_s], selected[:, idx_o]
        geom = self.geometry(selected_boxes, idx_s, idx_o)
        pair = self.pair_mlp(torch.cat([s_feat, o_feat, s_feat * o_feat,
                                         torch.zeros_like(s_feat), geom], dim=-1))
        pair = self.context(pair)
        ness_all = self.relationness(pair).squeeze(-1)
        k = min(top_k_pairs, pair.shape[1])
        top = ness_all.topk(k, dim=1).indices
        pair = torch.gather(pair, 1, top[..., None].expand(-1, -1, pair.shape[-1]))
        ness = torch.gather(ness_all, 1, top)
        s_idx = idx_s[None].expand(bsz, -1).gather(1, top)
        o_idx = idx_o[None].expand(bsz, -1).gather(1, top)
        z_rel = F.normalize(self.rel_proj(pair), dim=-1)
        output = {
            "pair_features": pair,
            "relationness_logits": ness,
            "z_rel": z_rel,
            "subject_slot": s_idx,
            "object_slot": o_idx,
            "selected_object_slots": chosen,
        }
        if relation_text is not None:
            relation_text = F.normalize(relation_text, dim=-1)
            output["relation_logits"] = torch.einsum("bkd,rd->bkr", z_rel, relation_text) / self.log_tau_rel.exp().clamp_min(1e-4)
        return output


@dataclass
class HOVRSGOutput:
    boxes: Tensor
    objectness_logits: Tensor
    object_scores: Tensor
    z_leaf: Tensor
    z_group: Tensor
    leaf_logits: Tensor
    group_logits: Tensor
    relations: Dict[str, Tensor]


class HOVRSG(nn.Module):
    def __init__(self, visual_dim: int, d_model: int, num_queries: int, d_latent: int = 512):
        super().__init__()
        if d_model % 8 != 0:
            raise ValueError("d_model must be divisible by 8 for the transformer heads")
        self.visual_dim = int(visual_dim)
        self.d_model = int(d_model)
        self.d_latent = int(d_latent)
        self.input_proj = nn.Linear(self.visual_dim, self.d_model)
        self.query_embed = nn.Embedding(num_queries, self.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=8, dim_feedforward=4 * self.d_model,
            batch_first=True, norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(layer, num_layers=6)
        self.box_head = MLP(self.d_model, self.d_model, 4, depth=3)
        self.objectness_head = nn.Linear(self.d_model, 1)
        self.object_head = HierarchicalPrototypeHead(self.d_model, self.d_latent)
        self.relation_head = SparseRelationDecoder(self.d_model, self.d_latent)

    def forward(
        self, visual_features: Tensor, leaf_text: Tensor, group_text: Tensor,
        relation_text: Optional[Tensor] = None, top_m: int = 100, top_k_pairs: int = 256,
    ) -> HOVRSGOutput:
        if visual_features.ndim != 3 or visual_features.shape[-1] != self.visual_dim:
            raise ValueError(
                f"Expected visual features [B, S, {self.visual_dim}], "
                f"got {tuple(visual_features.shape)}"
            )
        for name, prototype in (("leaf_text", leaf_text), ("group_text", group_text)):
            if prototype.ndim != 2 or prototype.shape[-1] != self.d_latent:
                raise ValueError(
                    f"{name} must have shape [classes, {self.d_latent}], "
                    f"got {tuple(prototype.shape)}"
                )
        if relation_text is not None and (relation_text.ndim != 2 or relation_text.shape[-1] != self.d_latent):
            raise ValueError(
                f"relation_text must have shape [predicates, {self.d_latent}], "
                f"got {tuple(relation_text.shape)}"
            )
        memory = self.input_proj(visual_features)
        bsz = memory.shape[0]
        queries = self.query_embed.weight[None].expand(bsz, -1, -1)
        slots = self.query_decoder(queries, memory)
        boxes = self.box_head(slots).sigmoid()
        objectness_logits = self.objectness_head(slots).squeeze(-1)
        object_scores = objectness_logits.sigmoid()
        obj = self.object_head(slots, leaf_text, group_text)
        relations = self.relation_head(slots, boxes, object_scores, relation_text, top_m, top_k_pairs)
        return HOVRSGOutput(
            boxes=boxes, objectness_logits=objectness_logits, object_scores=object_scores,
            z_leaf=obj["z_leaf"], z_group=obj["z_group"],
            leaf_logits=obj["leaf_logits"], group_logits=obj["group_logits"], relations=relations,
        )
