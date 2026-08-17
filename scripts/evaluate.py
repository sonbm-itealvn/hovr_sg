from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hovr_sg.data import UnifiedSceneGraphDataset, collate_scene_graph
from hovr_sg.models import HOVRSG, PrototypeBank, TinyImageEncoder, deterministic_text_embeddings
from hovr_sg.utils.config import load_yaml
from hovr_sg.utils.ontology import Ontology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    ontology = Ontology(args.ontology)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config", {})
    mc = cfg.get("model", {})
    dim = int(cfg.get("ontology", {}).get("text_dim", mc.get("d_latent", 256)))
    encoder = TinyImageEncoder(int(mc.get("visual_dim", 256))).to(device)
    model = HOVRSG(int(mc.get("visual_dim", 256)), int(mc.get("d_model", 256)), int(mc.get("num_queries", 64)), int(mc.get("d_latent", 256))).to(device)
    encoder.load_state_dict(checkpoint["encoder"]); model.load_state_dict(checkpoint["model"])
    encoder.eval(); model.eval()
    prototypes = PrototypeBank(
        deterministic_text_embeddings(ontology.leaf_names(), dim),
        deterministic_text_embeddings(ontology.group_names(), dim),
        deterministic_text_embeddings(ontology.predicate_names(), dim),
    ).to(device)
    ds = UnifiedSceneGraphDataset(args.jsonl, ontology, args.image_root, int(cfg.get("training", {}).get("image_size", 256)))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_scene_graph)
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluation"):
            visual = encoder(batch["images"].to(device))
            out = model(visual, prototypes.leaf, prototypes.groups, prototypes.relations,
                        int(mc.get("top_m_objects", 16)), int(mc.get("top_k_pairs", 64)))
            leaf_prob = out.leaf_logits.softmax(-1)[0]
            group_prob = out.group_logits.sigmoid()[0]
            obj_score, obj_label = leaf_prob.max(-1)
            rel_prob = out.relations.get("relation_logits")
            triplets = []
            if rel_prob is not None:
                rel_prob = rel_prob.softmax(-1)[0]
                for k in range(rel_prob.shape[0]):
                    r_score, r_label = rel_prob[k].max(-1)
                    s = int(out.relations["subject_slot"][0, k])
                    o = int(out.relations["object_slot"][0, k])
                    triplets.append({
                        "subject_slot": s, "predicate": ontology.predicate_names()[int(r_label)],
                        "object_slot": o, "score": float(r_score * out.relations["relationness_logits"][0, k].sigmoid()),
                    })
            predictions.append({
                "image_id": batch["samples"][0]["image_id"],
                "objects": [{"slot": i, "label": ontology.leaf_names()[int(obj_label[i])], "score": float(obj_score[i]),
                             "group_scores": group_prob[i].tolist(), "box": out.boxes[0, i].tolist()}
                            for i in range(len(obj_label)) if float(obj_score[i]) > 0.05],
                "relations": triplets,
            })
    output = {"num_images": len(predictions), "predictions": predictions,
              "ontology": ontology.version, "note": "Use project-specific COCO/SGG evaluator for AP/R@K."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
