from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hovr_sg.data import UnifiedSceneGraphDataset, collate_scene_graph
from hovr_sg.evaluation.predict import predict_dataset
from hovr_sg.models import (
    CLIPTextPrototypeEncoder,
    HOVRSG,
    PrototypeBank,
    build_image_encoder,
    deterministic_text_embeddings,
)
from hovr_sg.utils.ontology import Ontology


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_prototypes(cfg: dict, ontology: Ontology, device: torch.device, checkpoint: dict):
    model_cfg = cfg.get("model", {})
    saved = checkpoint.get("prototypes")
    if saved and all(key in saved for key in ("leaf", "groups", "relations")):
        return PrototypeBank(saved["leaf"], saved["groups"], saved["relations"]).to(device)

    backbone = str(model_cfg.get("backbone", "clip")).lower()
    if backbone in {"tiny", "tiny_cnn"}:
        dim = int(model_cfg.get("d_latent", cfg.get("ontology", {}).get("text_dim", 256)))
        values = (
            deterministic_text_embeddings(ontology.leaf_names(), dim),
            deterministic_text_embeddings(ontology.group_names(), dim),
            deterministic_text_embeddings(ontology.predicate_names(), dim),
        )
    else:
        encoder = CLIPTextPrototypeEncoder(
            model_name=str(model_cfg.get("backbone_name", "openai/clip-vit-base-patch32")),
            prompt_template=str(cfg.get("ontology", {}).get(
                "prompt_template", "a photo of a {label}"
            )),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
            device=device,
        )
        values = (
            encoder.encode(ontology.leaf_names()).cpu(),
            encoder.encode(ontology.group_names()).cpu(),
            encoder.encode(ontology.predicate_names()).cpu(),
        )
        if int(model_cfg.get("d_latent", encoder.output_dim)) != encoder.output_dim:
            raise ValueError("Checkpoint config d_latent does not match the text encoder projection width")
    return PrototypeBank(*values).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ontology = Ontology(args.ontology)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    mc = cfg.get("model", {})
    encoder = build_image_encoder(mc).to(device)
    resolved_visual_dim = int(checkpoint.get("resolved", {}).get("visual_dim", encoder.visual_dim))
    if encoder.visual_dim != resolved_visual_dim:
        raise ValueError(
            f"Backbone visual_dim={encoder.visual_dim} differs from checkpoint visual_dim={resolved_visual_dim}"
        )
    mc["visual_dim"] = resolved_visual_dim
    model = HOVRSG(
        resolved_visual_dim,
        int(mc.get("d_model", 256)),
        int(mc.get("num_queries", 64)),
        int(checkpoint.get("resolved", {}).get("text_dim", mc.get("d_latent", 256))),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    model.load_state_dict(checkpoint["model"])
    encoder.eval()
    model.eval()
    prototypes = build_prototypes(cfg, ontology, device, checkpoint)

    image_size = int(mc.get("image_size", cfg.get("training", {}).get("image_size", 224)))
    image_mean = tuple(mc.get("image_mean", [0.485, 0.456, 0.406]))
    image_std = tuple(mc.get("image_std", [0.229, 0.224, 0.225]))
    ds = UnifiedSceneGraphDataset(
        args.jsonl, ontology, args.image_root, image_size, image_mean, image_std
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_scene_graph)
    _, predictions, metrics = predict_dataset(
        encoder, model, prototypes, loader, device, ontology,
        top_m=int(mc.get("top_m_objects", 16)),
        top_k_pairs=int(mc.get("top_k_pairs", 64)),
        desc="evaluation",
    )
    output = {
        "num_images": len(predictions),
        "predictions": predictions,
        "metrics": metrics,
        "ontology": ontology.version,
        "note": "Object metrics are COCO-style AP; relation metrics are scene-graph Recall@K.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
