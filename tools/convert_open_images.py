from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from hovr_sg.data.adapters import canonical_object, canonical_relation
from hovr_sg.utils.ontology import Ontology


def read_csv(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def first(row: dict, names: List[str], default=""):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--boxes-csv", required=True)
    parser.add_argument("--relations-csv", required=False, default=None)
    parser.add_argument("--class-descriptions", required=False, default=None)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotation-scope", default="exhaustive")
    args = parser.parse_args()
    ontology = Ontology(args.ontology)
    class_names = {}
    if args.class_descriptions:
        for row in read_csv(args.class_descriptions):
            mid = first(row, ["LabelName", "ClassID", "MID"])
            name = first(row, ["DisplayName", "ClassName", "Name"], mid)
            class_names[mid] = name
    records: Dict[str, dict] = {}
    object_key_to_id: Dict[tuple, int] = {}
    next_id = defaultdict(int)
    for row in read_csv(args.boxes_csv):
        image_id = first(row, ["ImageID", "image_id"])
        label_id = first(row, ["ClassName", "LabelName", "label"])
        raw_label = class_names.get(label_id, label_id)
        mapped = canonical_object(raw_label, ontology)
        if not mapped:
            continue
        width = int(float(first(row, ["ImageWidth", "width"], "1")))
        height = int(float(first(row, ["ImageHeight", "height"], "1")))
        x1 = float(first(row, ["XMin", "xmin"], "0"))
        x2 = float(first(row, ["XMax", "xmax"], "0"))
        y1 = float(first(row, ["YMin", "ymin"], "0"))
        y2 = float(first(row, ["YMax", "ymax"], "0"))
        if max(x1, x2, y1, y2) <= 1.0:
            bbox = [x1 * width, y1 * height, x2 * width, y2 * height]
        else:
            bbox = [x1, y1, x2, y2]
        leaf, groups = mapped
        rec = records.setdefault(image_id, {
            "image_id": image_id, "image_path": str(Path(args.images_root) / f"{image_id}.jpg"),
            "width": width, "height": height, "objects": [], "relations": [],
            "annotation_scope": args.annotation_scope, "negative_labels": [],
            "metadata": {"source": "open_images"},
        })
        oid = next_id[image_id]
        next_id[image_id] += 1
        object_key_to_id[(image_id, len(rec["objects"]))] = oid
        rec["objects"].append({"id": oid, "bbox": bbox, "label": leaf, "group_labels": groups,
                                "attributes": [], "is_group": row.get("IsGroupOf", "") == "1",
                                "source": "open_images"})
    if args.relations_csv:
        for row in read_csv(args.relations_csv):
            image_id = first(row, ["ImageID", "image_id"])
            if image_id not in records:
                continue
            pred = first(row, ["RelationshipLabel", "Predicate", "relation", "LabelName"])
            mapped_rel = canonical_relation(pred, ontology)
            if mapped_rel is None:
                continue
            subject_index = int(first(row, ["SubjectIndex", "subject_index"], "-1"))
            object_index = int(first(row, ["ObjectIndex", "object_index"], "-1"))
            if not (0 <= subject_index < len(records[image_id]["objects"]) and 0 <= object_index < len(records[image_id]["objects"])):
                continue
            pred_name, pred_group = mapped_rel
            objects = records[image_id]["objects"]
            records[image_id]["relations"].append({
                "subject_id": objects[subject_index]["id"],
                "object_id": objects[object_index]["id"],
                "predicate": pred_name, "predicate_group": pred_group, "source": "open_images",
            })
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for record in records.values():
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
