"""Hungarian assignment for DETR-style object queries."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


class HungarianMatcher:
    """Match predicted queries to scene-graph objects with one-to-one assignment.

    The matching cost is intentionally detached from autograd.  Gradients are
    computed by the subsequent matched losses, as in DETR.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_objectness: float = 1.0,
    ):
        if cost_class == 0 and cost_bbox == 0 and cost_objectness == 0:
            raise ValueError("At least one Hungarian matching cost must be non-zero")
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_objectness = float(cost_objectness)

    @torch.no_grad()
    def __call__(self, outputs, samples: List[dict]) -> List[Tuple[Tensor, Tensor]]:
        """Return per-image ``(query_indices, target_indices)`` tensors."""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover - dependency configuration
            raise ImportError(
                "HungarianMatcher requires scipy. Install the project dependencies with "
                "`pip install -e .`."
            ) from exc

        probabilities = outputs.leaf_logits.softmax(-1)
        objectness = outputs.objectness_logits.sigmoid()
        matches: List[Tuple[Tensor, Tensor]] = []
        for batch_index, sample in enumerate(samples):
            max_targets = outputs.boxes.shape[1]
            target_labels = sample["leaf_indices"][:max_targets].to(outputs.leaf_logits.device)
            target_boxes = sample["boxes"][:max_targets].to(outputs.boxes.device)
            if target_labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=outputs.boxes.device)
                matches.append((empty, empty))
                continue
            class_cost = -probabilities[batch_index][:, target_labels]
            bbox_cost = torch.cdist(outputs.boxes[batch_index], target_boxes, p=1)
            objectness_cost = -objectness[batch_index][:, None].expand(-1, target_labels.numel())
            cost = (
                self.cost_class * class_cost
                + self.cost_bbox * bbox_cost
                + self.cost_objectness * objectness_cost
            )
            query_idx, target_idx = linear_sum_assignment(cost.float().cpu().numpy())
            matches.append((
                torch.as_tensor(query_idx, dtype=torch.long, device=outputs.boxes.device),
                torch.as_tensor(target_idx, dtype=torch.long, device=outputs.boxes.device),
            ))
        return matches
