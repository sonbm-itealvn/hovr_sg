from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from hovr_sg.data.adapters import canonical_object, canonical_relation
from hovr_sg.utils.ontology import Ontology


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--image-data", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--relationships", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ontology = Ontology(args.ontology)
    image_data = {str(x["image_id"]): x for x in load_json(args.image_data)}
    objects = {str(x["image_id"]): x for x in load_json(args.objects)}
    relations = {str(x["image_id"]): x for x in load_json(args.relationships)}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for image_id, meta in image_data.items():
            width, height = int(meta.get("width", 0)), int(meta.get("height", 0))
            obj_items = objects.get(image_id, {}).get("objects", [])
            object_rows = []
            valid_ids = set()
            for obj in obj_items:
                labels = obj.get("names") or [obj.get("name", "")]
                mapped = None
                for label in labels:
                    mapped = canonical_object(label, ontology)
                    if mapped:
                        break
                if mapped is None:
                    continue
                leaf, groups = mapped
                oid = int(obj["object_id"])
                x, y = float(obj.get("x", 0)), float(obj.get("y", 0))
                w, h = float(obj.get("w", 0)), float(obj.get("h", 0))
                object_rows.append({
                    "id": oid, "bbox": [x, y, x + w, y + h], "label": leaf,
                    "group_labels": groups, "attributes": obj.get("attributes", []),
                    "is_group": False, "source": "visual_genome",
                })
                valid_ids.add(oid)
            relation_rows = []
            for rel in relations.get(image_id, {}).get("relationships", []):
                subj = int(rel.get("subject", {}).get("object_id", -1))
                obj = int(rel.get("object", {}).get("object_id", -1))
                if subj not in valid_ids or obj not in valid_ids:
                    continue
                mapped_rel = canonical_relation(rel.get("predicate", ""), ontology)
                if mapped_rel is None:
                    continue
                pred, pred_group = mapped_rel
                relation_rows.append({
                    "subject_id": subj, "object_id": obj, "predicate": pred,
                    "predicate_group": pred_group, "source": "visual_genome",
                })
            record = {
                "image_id": image_id,
                "image_path": str(Path(args.images_root) / f"{image_id}.jpg"),
                "width": width, "height": height,
                "objects": object_rows, "relations": relation_rows,
                "annotation_scope": "partial", "negative_labels": [],
                "metadata": {"source": "visual_genome"},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
