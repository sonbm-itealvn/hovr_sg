from __future__ import annotations

from typing import Iterable, List

import torch
from tqdm import tqdm

from .metrics import evaluate_scene_graph


def decode_predictions(out, sample: dict, ontology) -> dict:
    leaf_prob = out.leaf_logits.softmax(-1)[0]
    group_prob = out.group_logits.sigmoid()[0]
    object_score, object_label = leaf_prob.max(-1)
    relations = []
    relation_prob = out.relations.get("relation_logits")
    if relation_prob is not None:
        relation_prob = relation_prob.softmax(-1)[0]
        for index in range(relation_prob.shape[0]):
            predicate_score, predicate_label = relation_prob[index].max(-1)
            relations.append({
                "subject_slot": int(out.relations["subject_slot"][0, index]),
                "predicate": ontology.predicate_names()[int(predicate_label)],
                "object_slot": int(out.relations["object_slot"][0, index]),
                "score": float(
                    predicate_score * out.relations["relationness_logits"][0, index].sigmoid()
                ),
            })
    return {
        "image_id": sample["image_id"],
        "objects": [
            {
                "slot": index,
                "label": ontology.leaf_names()[int(object_label[index])],
                "score": float(object_score[index]),
                "group_scores": group_prob[index].tolist(),
                "box": out.boxes[0, index].tolist(),
            }
            for index in range(len(object_label)) if float(object_score[index]) > 0.05
        ],
        "relations": relations,
    }


def predict_dataset(
    encoder, model, prototypes, loader, device, ontology,
    top_m: int = 16, top_k_pairs: int = 64, desc: str = "evaluation",
):
    records: List[dict] = []
    predictions: List[dict] = []
    encoder.eval()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            visual = encoder(batch["images"].to(device))
            out = model(
                visual, prototypes.leaf, prototypes.groups, prototypes.relations,
                top_m=int(top_m),
                top_k_pairs=int(top_k_pairs),
            )
            records.append(batch["samples"][0])
            predictions.append(decode_predictions(out, batch["samples"][0], ontology))
    return records, predictions, evaluate_scene_graph(records, predictions, ontology)
