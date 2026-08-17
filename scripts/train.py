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
    batch = len(samples)
    objectness = torch.zeros(batch, num_queries, device=device)
    boxes = torch.zeros(batch, num_queries, 4, device=device)
    leaf = torch.zeros(batch, num_queries, dtype=torch.long, device=device)
    group = torch.zeros(batch, num_queries, group_count, device=device)
    assignments: List[Dict[int, int]] = []
    for b, sample in enumerate(samples):
        assignment = {}
        for q, obj in enumerate(sample["boxes"][:num_queries].to(device)):
            objectness[b, q] = 1.0
            boxes[b, q] = obj
            label = int(sample["leaf_indices"][q])
            leaf[b, q] = label
            for g in sample["group_indices"][q]:
                group[b, q, int(g)] = 1.0
            assignment[int(sample["object_ids"][q])] = q
        assignments.append(assignment)
    return {
        "objectness": objectness,
        "boxes": boxes,
        "leaf": leaf,
        "group": group,
        "assignments": assignments,
    }


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


def compute_loss(out, targets, prototypes: PrototypeBank, ontology: Ontology, weights: dict):
    pos = targets["objectness"] > 0.5
    objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out.objectness_logits, targets["objectness"]
    )
    if pos.any():
        leaf_loss = torch.nn.functional.cross_entropy(out.leaf_logits[pos], targets["leaf"][pos])
        group_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out.group_logits[pos], targets["group"][pos]
        )
        box_loss = torch.nn.functional.l1_loss(out.boxes[pos], targets["boxes"][pos])
        ancestor_loss = ancestor_consistency(
            out.leaf_logits[pos], out.group_logits[pos], ontology.leaf_to_groups
        )
        sibling_losses = []
        for q_label, info in ontology.object_leaves.items():
            if not info.siblings:
                continue
            idx = ontology.leaf_index(q_label)
            mask = pos & (targets["leaf"] == idx)
            if not mask.any():
                continue
            pos_score = out.z_leaf[mask] @ prototypes.leaf[idx]
            for sibling in info.siblings:
                if sibling in ontology.leaf_to_idx:
                    neg_score = out.z_leaf[mask] @ prototypes.leaf[ontology.leaf_index(sibling)]
                    sibling_losses.append(torch.relu(0.20 - pos_score + neg_score).mean())
        sibling_loss = torch.stack(sibling_losses).mean() if sibling_losses else out.z_leaf.sum() * 0.0
    else:
        leaf_loss = group_loss = box_loss = ancestor_loss = sibling_loss = out.z_leaf.sum() * 0.0

    relationness = out.relations["relationness_logits"]
    relation_targets = torch.zeros_like(relationness)
    predicate_targets = torch.full(
        relationness.shape, -1, dtype=torch.long, device=relationness.device
    )
    for b, sample in enumerate(targets["assignments"]):
        query_for_object = sample
        selected_queries = out.relations["selected_object_slots"][b]
        pair_s = out.relations["subject_slot"][b]
        pair_o = out.relations["object_slot"][b]
        for rel in targets["raw_samples"][b]["relations"]:
            sq = query_for_object.get(rel["subject_id"])
            oq = query_for_object.get(rel["object_id"])
            if sq is None or oq is None:
                continue
            selected_s = (selected_queries == sq).nonzero(as_tuple=False)
            selected_o = (selected_queries == oq).nonzero(as_tuple=False)
            if not len(selected_s) or not len(selected_o):
                continue
            ps = int(selected_s[0])
            po = int(selected_o[0])
            hits = ((pair_s == ps) & (pair_o == po)).nonzero(as_tuple=False)
            if len(hits):
                k = int(hits[0])
                relation_targets[b, k] = 1.0
                predicate_targets[b, k] = int(rel["predicate_index"])
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
        args.train_jsonl, ontology, args.image_root, image_size, image_mean, image_std
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
    trainable_params = list(model.parameters()) + list(prototypes.parameters())
    trainable_params += [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.get("training", {}).get("lr", 1e-4)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 1e-4)),
    )
    weights = cfg.get("loss", {})
    epochs = args.epochs or int(cfg.get("training", {}).get("epochs", 10))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        encoder.train()
        model.train()
        prototypes.train()
        running = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in progress:
            images = batch["images"].to(device)
            targets = make_targets(
                batch["samples"], ontology, int(model_cfg.get("num_queries", 64)),
                device, len(ontology.group_names()),
            )
            targets["raw_samples"] = batch["samples"]
            visual = encoder(images)
            out = model(
                visual, prototypes.leaf, prototypes.groups, prototypes.relations,
                int(model_cfg.get("top_m_objects", 16)), int(model_cfg.get("top_k_pairs", 64)),
            )
            terms = compute_loss(out, targets, prototypes, ontology, weights)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
            optimizer.step()
            running += float(terms["total"].detach())
            progress.set_postfix(loss=f"{running / max(progress.n, 1):.4f}")
        checkpoint = {
            "epoch": epoch + 1,
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
        json.dumps({"epochs": epochs, "device": str(device), "config": cfg}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
