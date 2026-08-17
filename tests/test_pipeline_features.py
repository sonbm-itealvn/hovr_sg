from types import SimpleNamespace

import torch

from hovr_sg.losses import HungarianMatcher
from hovr_sg.models.hovr_sg import SparseRelationDecoder
from scripts.train import build_stage_schedule


def test_hungarian_matcher_is_one_to_one():
    outputs = SimpleNamespace(
        leaf_logits=torch.tensor([[[5.0, 0.0], [0.0, 5.0], [4.0, 1.0]]]),
        objectness_logits=torch.tensor([[4.0, 4.0, -2.0]]),
        boxes=torch.tensor([[[0.05, 0.05, 0.25, 0.25], [0.65, 0.65, 0.9, 0.9], [0.0, 0.0, 0.1, 0.1]]]),
    )
    samples = [{
        "leaf_indices": torch.tensor([0, 1]),
        "boxes": torch.tensor([[0.0, 0.0, 0.25, 0.25], [0.65, 0.65, 0.9, 0.9]]),
    }]
    query_indices, target_indices = HungarianMatcher()(outputs, samples)[0]
    assert len(query_indices) == 2
    assert len(set(query_indices.tolist())) == 2
    assert sorted(target_indices.tolist()) == [0, 1]


def test_union_region_features_are_pooled_from_memory():
    memory = torch.arange(16.0).view(1, 4, 4)
    boxes = torch.tensor([[[0.0, 0.0, 0.6, 0.6], [0.4, 0.4, 1.0, 1.0]]])
    subject = torch.tensor([0])
    object_ = torch.tensor([1])
    pooled = SparseRelationDecoder.union_region_features(memory, boxes, subject, object_)
    assert pooled.shape == (1, 1, 4)
    assert pooled.abs().sum() > 0


def test_stage_schedule_uses_all_configured_stages():
    config = {"stages": {
        "detector_warmup_epochs": 1,
        "hierarchical_epochs": 1,
        "relation_epochs": 1,
        "joint_epochs": 1,
    }}
    assert build_stage_schedule(config, 4) == [
        ("detector_warmup", 1), ("hierarchical", 1), ("relation", 1), ("joint", 1)
    ]
