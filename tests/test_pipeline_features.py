from types import SimpleNamespace

import torch

from hovr_sg.losses import HungarianMatcher
from hovr_sg.models.hovr_sg import SparseRelationDecoder
from hovr_sg.evaluation.metrics import scene_graph_metrics
from scripts.train import atomic_step, build_stage_schedule, finite_nested


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


def test_stage_schedule_can_extend_last_stage_for_resume():
    config = {"stages": {
        "detector_warmup_epochs": 2,
        "hierarchical_epochs": 2,
        "relation_epochs": 3,
        "joint_epochs": 3,
    }}
    assert build_stage_schedule(config, 20, extend_last_stage=True) == [
        ("detector_warmup", 2), ("hierarchical", 2),
        ("relation", 3), ("joint", 13),
    ]


def test_atomic_step_rejects_nonfinite_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, eps=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    parameter.grad = torch.tensor([float("nan")])
    assert atomic_step(optimizer, scaler, [parameter], 0.25) is False
    assert parameter.item() == 1.0


def test_checkpoint_finite_validator_rejects_nan():
    assert finite_nested({"state": torch.ones(2), "metadata": [1, "ok"]})
    assert not finite_nested({"state": torch.tensor([float("nan")])})


def test_relation_metric_uses_global_query_slots():
    ontology = SimpleNamespace(
        leaf_to_idx={"man": 0, "cup": 1},
        predicate_to_idx={"holding": 0},
        leaf_names=lambda: ["man", "cup"],
        predicate_names=lambda: ["holding"],
        leaf_index=lambda label: {"man": 0, "cup": 1}[label],
        predicate_index=lambda label: {"holding": 0}[label],
    )
    records = [{
        "object_ids": [10, 20],
        "boxes": torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]),
        "leaf_indices": torch.tensor([0, 1]),
        "relations": [{"subject_id": 10, "object_id": 20, "predicate_index": 0}],
    }]
    predictions = [{
        "objects": [
            {"slot": 7, "label": "man", "box": [0.0, 0.0, 0.5, 0.5]},
            {"slot": 3, "label": "cup", "box": [0.5, 0.5, 1.0, 1.0]},
        ],
        "relations": [{"subject_slot": 7, "object_slot": 3, "predicate": "holding", "score": 1.0}],
    }]
    metrics = scene_graph_metrics(records, predictions, ontology)
    assert metrics["relation_Recall@50"] == 1.0
