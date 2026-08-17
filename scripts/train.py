from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hovr_sg.data import UnifiedSceneGraphDataset, collate_scene_graph
from hovr_sg.losses import HungarianMatcher
from hovr_sg.losses.hovr_losses import ancestor_consistency
from hovr_sg.models import (
    CLIPTextPrototypeEncoder,
    HOVRSG,
    PrototypeBank,
    build_image_encoder,
    deterministic_text_embeddings,
)
from hovr_sg.utils.config import load_yaml
from hovr_sg.utils.ontology import Ontology
from hovr_sg.utils.seed import seed_everything


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def make_targets(
    samples: List[dict], ontology: Ontology, num_queries: int,
    device: torch.device, group_count: int,
):
    """Create padded tensors; query assignment is deliberately deferred to DETR matching."""
    batch = len(samples)
    boxes = torch.zeros(batch, num_queries, 4, device=device)
    leaf = torch.zeros(batch, num_queries, dtype=torch.long, device=device)
    group = torch.zeros(batch, num_queries, group_count, device=device)
    for b, sample in enumerate(samples):
        count = min(len(sample["boxes"]), num_queries)
        if count == 0:
            continue
        boxes[b, :count] = sample["boxes"][:count].to(device)
        leaf[b, :count] = sample["leaf_indices"][:count].to(device)
        for target_index, groups in enumerate(sample["group_indices"][:count]):
            for group_index in groups:
                group[b, target_index, int(group_index)] = 1.0
    return {"boxes": boxes, "leaf": leaf, "group": group}


def matched_indices(matches, device: torch.device):
    query_indices = [pair[0].to(device) for pair in matches if pair[0].numel()]
    target_indices = [pair[1].to(device) for pair in matches if pair[1].numel()]
    if not query_indices:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.cat(query_indices), torch.cat(target_indices)


def build_prototypes(cfg: dict, ontology: Ontology, device: torch.device):
    """Build text prototypes and return their resolved shared embedding width."""
    model_cfg = cfg.setdefault("model", {})
    backbone = str(model_cfg.get("backbone", "clip")).lower()
    if backbone in {"tiny", "tiny_cnn"}:
        dim = int(model_cfg.get("d_latent", cfg.get("ontology", {}).get("text_dim", 256)))
        leaf = deterministic_text_embeddings(ontology.leaf_names(), dim)
        groups = deterministic_text_embeddings(ontology.group_names(), dim)
        relations = deterministic_text_embeddings(ontology.predicate_names(), dim)
    else:
        text_encoder = CLIPTextPrototypeEncoder(
            model_name=str(model_cfg.get("backbone_name", "openai/clip-vit-base-patch32")),
            prompt_template=str(cfg.get("ontology", {}).get(
                "prompt_template", "a photo of a {label}"
            )),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
            device=device,
        )
        leaf = text_encoder.encode(ontology.leaf_names()).cpu()
        groups = text_encoder.encode(ontology.group_names()).cpu()
        relations = text_encoder.encode(ontology.predicate_names()).cpu()
        dim = text_encoder.output_dim
        configured_dim = model_cfg.get("d_latent")
        if configured_dim is not None and int(configured_dim) != dim:
            raise ValueError(
                f"model.d_latent={configured_dim} does not match the pretrained text "
                f"projection_dim={dim}. Set d_latent to {dim} or omit it."
            )
    cfg.setdefault("ontology", {})["text_dim"] = int(dim)
    model_cfg["d_latent"] = int(dim)
    bank = PrototypeBank(
        leaf, groups, relations,
        trainable=bool(cfg.get("ontology", {}).get("trainable_prototypes", False)),
    ).to(device)
    return bank


def compute_loss(out, targets, matches, prototypes: PrototypeBank, ontology: Ontology, weights: dict):
    device = out.boxes.device
    objectness_target = torch.zeros_like(out.objectness_logits)
    matched_rows = []
    for batch_index, (query_indices, target_indices) in enumerate(matches):
        if query_indices.numel() == 0:
            continue
        query_indices = query_indices.to(device)
        target_indices = target_indices.to(device)
        objectness_target[batch_index, query_indices] = 1.0
        matched_rows.append((batch_index, query_indices, target_indices))

    objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out.objectness_logits, objectness_target
    )
    if matched_rows:
        pred_boxes = torch.cat([out.boxes[b, q] for b, q, _ in matched_rows])
        target_boxes = torch.cat([targets["boxes"][b, t] for b, _, t in matched_rows])
        pred_leaf = torch.cat([out.leaf_logits[b, q] for b, q, _ in matched_rows])
        target_leaf = torch.cat([targets["leaf"][b, t] for b, _, t in matched_rows])
        pred_group = torch.cat([out.group_logits[b, q] for b, q, _ in matched_rows])
        target_group = torch.cat([targets["group"][b, t] for b, _, t in matched_rows])
        z_leaf = torch.cat([out.z_leaf[b, q] for b, q, _ in matched_rows])
        leaf_loss = torch.nn.functional.cross_entropy(pred_leaf, target_leaf)
        group_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_group, target_group)
        box_loss = torch.nn.functional.l1_loss(pred_boxes, target_boxes)
        ancestor_loss = ancestor_consistency(pred_leaf, pred_group, ontology.leaf_to_groups)
        sibling_losses = []
        for label_index, label_info in ontology.object_leaves.items():
            if not label_info.siblings:
                continue
            label_id = ontology.leaf_index(label_index)
            mask = target_leaf == label_id
            if not mask.any():
                continue
            positive = z_leaf[mask] @ prototypes.leaf[label_id]
            for sibling in label_info.siblings:
                if sibling in ontology.leaf_to_idx:
                    negative = z_leaf[mask] @ prototypes.leaf[ontology.leaf_index(sibling)]
                    sibling_losses.append(torch.relu(0.20 - positive + negative).mean())
        sibling_loss = torch.stack(sibling_losses).mean() if sibling_losses else z_leaf.sum() * 0.0
    else:
        zero = out.z_leaf.sum() * 0.0
        leaf_loss = group_loss = box_loss = ancestor_loss = sibling_loss = zero

    relationness = out.relations["relationness_logits"]
    relation_targets = torch.zeros_like(relationness)
    predicate_targets = torch.full(
        relationness.shape, -1, dtype=torch.long, device=relationness.device
    )
    for batch_index, sample in enumerate(targets["raw_samples"]):
        query_for_object = {}
        if batch_index < len(matches):
            query_indices, target_indices = matches[batch_index]
            for query_index, target_index in zip(query_indices.tolist(), target_indices.tolist()):
                if target_index < len(sample["object_ids"]):
                    query_for_object[int(sample["object_ids"][target_index])] = int(query_index)
        selected_queries = out.relations["selected_object_slots"][batch_index]
        pair_s = out.relations["subject_slot"][batch_index]
        pair_o = out.relations["object_slot"][batch_index]
        for rel in sample["relations"]:
            subject_query = query_for_object.get(int(rel["subject_id"]))
            object_query = query_for_object.get(int(rel["object_id"]))
            if subject_query is None or object_query is None:
                continue
            subject_local = (selected_queries == subject_query).nonzero(as_tuple=False)
            object_local = (selected_queries == object_query).nonzero(as_tuple=False)
            if not len(subject_local) or not len(object_local):
                continue
            hits = ((pair_s == int(subject_local[0])) & (pair_o == int(object_local[0]))).nonzero(as_tuple=False)
            if len(hits):
                relation_index = int(hits[0])
                relation_targets[batch_index, relation_index] = 1.0
                predicate_targets[batch_index, relation_index] = int(rel["predicate_index"])
    relation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        relationness, relation_targets
    )
    pred_loss = relation_loss * 0.0
    if "relation_logits" in out.relations and (predicate_targets >= 0).any():
        valid = predicate_targets >= 0
        pred_loss = torch.nn.functional.cross_entropy(
            out.relations["relation_logits"][valid], predicate_targets[valid]
        )
    terms = {
        "objectness": objectness_loss,
        "leaf": leaf_loss,
        "group": group_loss,
        "ancestor": ancestor_loss,
        "box_l1": box_loss,
        "sibling": sibling_loss,
        "relationness": relation_loss,
        "predicate": pred_loss,
    }
    total = sum(weights.get(k, 1.0) * value for k, value in terms.items())
    terms["total"] = total
    return terms


def build_stage_schedule(cfg: dict, total_epochs: int, override: int | None = None):
    configured = [
        ("detector_warmup", int(cfg.get("stages", {}).get("detector_warmup_epochs", 0))),
        ("hierarchical", int(cfg.get("stages", {}).get("hierarchical_epochs", 0))),
        ("relation", int(cfg.get("stages", {}).get("relation_epochs", 0))),
        ("joint", int(cfg.get("stages", {}).get("joint_epochs", 0))),
    ]
    if override is not None:
        remaining = int(override)
        schedule = []
        for name, count in configured:
            if remaining <= 0:
                break
            use = min(count, remaining)
            if use:
                schedule.append((name, use))
                remaining -= use
        if remaining:
            schedule.append(("joint", remaining))
        return schedule
    schedule = [(name, count) for name, count in configured if count > 0]
    scheduled_epochs = sum(count for _, count in schedule)
    if scheduled_epochs != total_epochs:
        schedule.append(("joint", max(0, total_epochs - scheduled_epochs)))
    return [(name, count) for name, count in schedule if count > 0]


def set_module_grad(module, enabled: bool):
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def configure_stage(stage: str, encoder, model, model_cfg: dict):
    """Apply the staged freeze/unfreeze policy described in the research config."""
    set_module_grad(encoder, False)
    set_module_grad(model, False)
    detector_modules = [
        model.query_embed, model.query_decoder, model.box_head,
        model.objectness_head, model.object_head,
    ]
    if stage in {"detector_warmup", "hierarchical"}:
        for module in detector_modules:
            set_module_grad(module, True)
    elif stage == "relation":
        set_module_grad(model.relation_head, True)
    elif stage == "joint":
        set_module_grad(model, True)
        if bool(model_cfg.get("train_backbone", False)) or int(model_cfg.get("unfreeze_last_n_layers", 0)) > 0:
            set_module_grad(encoder, True)
    else:
        raise ValueError(f"Unknown training stage: {stage}")


def stage_loss_weights(stage: str, weights: dict) -> dict:
    stage_weights = dict(weights)
    if stage == "detector_warmup":
        stage_weights.update({"ancestor": 0.0, "sibling": 0.0, "relationness": 0.0, "predicate": 0.0})
    elif stage == "hierarchical":
        stage_weights.update({"relationness": 0.0, "predicate": 0.0})
    elif stage == "relation":
        stage_weights.update({
            "objectness": 0.0, "leaf": 0.0, "group": 0.0, "ancestor": 0.0,
            "box_l1": 0.0, "sibling": 0.0,
        })
    return stage_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=False)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backbone", default=None, choices=["clip", "pretrained_vlm", "tiny_cnn"])
    parser.add_argument("--backbone-name", default=None)
    parser.add_argument("--train-backbone", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg = copy.deepcopy(cfg)
    model_cfg = cfg.setdefault("model", {})
    if args.backbone:
        model_cfg["backbone"] = args.backbone
    if args.backbone_name:
        model_cfg["backbone_name"] = args.backbone_name
    if args.train_backbone:
        model_cfg["train_backbone"] = True

    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(args.device or cfg.get("training", {}).get("device", "auto"))
    ontology = Ontology(args.ontology)

    encoder = build_image_encoder(model_cfg).to(device)
    model_cfg["visual_dim"] = int(encoder.visual_dim)
    if str(model_cfg.get("backbone", "clip")).lower() not in {"tiny", "tiny_cnn"}:
        model_cfg["image_size"] = int(encoder.image_size)
        model_cfg["image_mean"] = list(encoder.image_mean)
        model_cfg["image_std"] = list(encoder.image_std)
    image_size = int(model_cfg.get("image_size", cfg.get("training", {}).get("image_size", 224)))
    image_mean = tuple(model_cfg.get("image_mean", [0.485, 0.456, 0.406]))
    image_std = tuple(model_cfg.get("image_std", [0.229, 0.224, 0.225]))
    train_ds = UnifiedSceneGraphDataset(
        args.train_jsonl, ontology, args.image_root, image_size, image_mean, image_std,
        train=True, augmentation=cfg.get("augmentation", {}),
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("training", {}).get("batch_size", 2)),
        shuffle=True,
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
        collate_fn=collate_scene_graph,
    )

    prototypes = build_prototypes(cfg, ontology, device)
    model = HOVRSG(
        int(model_cfg["visual_dim"]), int(model_cfg.get("d_model", 256)),
        int(model_cfg.get("num_queries", 64)), int(model_cfg["d_latent"]),
    ).to(device)
    weights = cfg.get("loss", {})
    matcher_cfg = cfg.get("matching", {})
    matcher = HungarianMatcher(
        cost_class=float(matcher_cfg.get("cost_class", 1.0)),
        cost_bbox=float(matcher_cfg.get("cost_bbox", 5.0)),
        cost_objectness=float(matcher_cfg.get("cost_objectness", 1.0)),
    )
    epochs = args.epochs or int(cfg.get("training", {}).get("epochs", 10))
    stage_schedule = build_stage_schedule(cfg, epochs, args.epochs)
    amp_requested = bool(cfg.get("training", {}).get("amp", False))
    amp_enabled = amp_requested and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch < 2.4 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_epoch = 0

    for stage_name, stage_epochs in stage_schedule:
        configure_stage(stage_name, encoder, model, model_cfg)
        trainable_params = [
            parameter for module in (encoder, model, prototypes)
            for parameter in module.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(cfg.get("training", {}).get("lr", 1e-4)),
            weight_decay=float(cfg.get("training", {}).get("weight_decay", 1e-4)),
        )
        active_weights = stage_loss_weights(stage_name, weights)
        for _ in range(stage_epochs):
            current_epoch += 1
            encoder.train()
            model.train()
            prototypes.train()
            running = 0.0
            progress = tqdm(loader, desc=f"{stage_name} {current_epoch}/{epochs}")
            for batch in progress:
                images = batch["images"].to(device)
                targets = make_targets(
                    batch["samples"], ontology, int(model_cfg.get("num_queries", 64)),
                    device, len(ontology.group_names()),
                )
                targets["raw_samples"] = batch["samples"]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    visual = encoder(images)
                    out = model(
                        visual, prototypes.leaf, prototypes.groups, prototypes.relations,
                        top_m=int(model_cfg.get("top_m_objects", 16)),
                        top_k_pairs=int(model_cfg.get("top_k_pairs", 64)),
                    )
                    matches = matcher(out, batch["samples"])
                    terms = compute_loss(out, targets, matches, prototypes, ontology, active_weights)
                if scaler.is_enabled():
                    scaler.scale(terms["total"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    terms["total"].backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                    optimizer.step()
                running += float(terms["total"].detach())
                progress.set_postfix(stage=stage_name, loss=f"{running / max(progress.n, 1):.4f}")
            checkpoint = {
                "epoch": current_epoch,
                "stage": stage_name,
                "stage_schedule": stage_schedule,
                "amp_enabled": amp_enabled,
                "encoder": encoder.state_dict(),
                "model": model.state_dict(),
                "prototypes": {name: tensor.detach().cpu() for name, tensor in prototypes.state_dict().items()},
                "ontology": ontology.version,
                "config": cfg,
                "resolved": {
                    "visual_dim": int(model_cfg["visual_dim"]),
                    "text_dim": int(model_cfg["d_latent"]),
                    "backbone": str(model_cfg.get("backbone", "clip")),
                    "backbone_name": model_cfg.get("backbone_name"),
                },
            }
            torch.save(checkpoint, out_dir / "last.pt")
    (out_dir / "training_summary.json").write_text(
        json.dumps({
            "epochs": epochs, "device": str(device), "amp_enabled": amp_enabled,
            "stage_schedule": stage_schedule, "config": cfg,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
