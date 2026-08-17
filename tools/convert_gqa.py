from __future__ import annotations

import argparse
import json
from pathlib import Path

from hovr_sg.data.adapters import canonical_object, canonical_relation
from hovr_sg.utils.ontology import Ontology


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-graphs", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ontology = Ontology(args.ontology)
    graphs = load_json(args.scene_graphs)
    if isinstance(graphs, list):
        graphs = {str(x.get("imageId", x.get("image_id"))): x for x in graphs}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for image_id, graph in graphs.items():
            objects = []
            id_map = {}
            raw_objects = graph.get("objects", graph.get("nodes", {}))
            items = raw_objects.items() if isinstance(raw_objects, dict) else enumerate(raw_objects)
            for raw_id, raw in items:
                names = raw.get("name") or raw.get("names") or raw.get("label") or ""
                if isinstance(names, list):
                    names = names[0] if names else ""
                mapped = canonical_object(names, ontology)
                if mapped is None:
                    continue
                leaf, groups = mapped
                bbox = raw.get("bbox", raw.get("box", {}))
                if isinstance(bbox, dict):
                    x, y = float(bbox.get("x", 0)), float(bbox.get("y", 0))
                    w, h = float(bbox.get("w", bbox.get("width", 0))), float(bbox.get("h", bbox.get("height", 0)))
                    bbox = [x, y, x + w, y + h]
                oid = len(objects)
                id_map[str(raw_id)] = oid
                objects.append({
                    "id": oid, "bbox": [float(v) for v in bbox], "label": leaf,
                    "group_labels": groups, "attributes": raw.get("attributes", []),
                    "is_group": False, "source": "gqa",
                })
            relations = []
            for raw_id, raw in (raw_objects.items() if isinstance(raw_objects, dict) else enumerate(raw_objects)):
                source_id = id_map.get(str(raw_id))
                if source_id is None:
                    continue
                raw_relations = raw.get("relations", raw.get("relationships", []))
                if isinstance(raw_relations, dict):
                    raw_relations = [{"object": k, "name": v} for k, v in raw_relations.items()]
                for rel in raw_relations:
                    target_id = id_map.get(str(rel.get("object", rel.get("object_id", rel.get("target", "")))))
                    if target_id is None:
                        continue
                    mapped_rel = canonical_relation(rel.get("name", rel.get("predicate", rel.get("relation", ""))), ontology)
                    if mapped_rel is None:
                        continue
                    pred, group = mapped_rel
                    relations.append({"subject_id": source_id, "object_id": target_id,
                                      "predicate": pred, "predicate_group": group, "source": "gqa"})
            record = {
                "image_id": str(image_id),
                "image_path": str(Path(args.images_root) / f"{image_id}.jpg"),
                "width": int(graph.get("width", 0)), "height": int(graph.get("height", 0)),
                "objects": objects, "relations": relations,
                "annotation_scope": "partial", "negative_labels": [],
                "metadata": {"source": "gqa"},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
