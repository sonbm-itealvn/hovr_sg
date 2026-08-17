from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from hovr_sg.utils.ontology import Ontology


def read_records(paths: Iterable[str]) -> List[dict]:
    records = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--object-novel-ratio", type=float, default=0.2)
    parser.add_argument("--relation-novel-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    ontology = Ontology(args.ontology)
    records = read_records(args.input)
    object_count = Counter(obj["label"] for rec in records for obj in rec.get("objects", []))
    relation_count = Counter(rel["predicate"] for rec in records for rel in rec.get("relations", []))
    object_labels = [x for x in ontology.leaf_names() if object_count[x] > 0]
    relation_labels = [x for x in ontology.predicate_names() if relation_count[x] > 0]
    random.shuffle(object_labels)
    random.shuffle(relation_labels)
    n_obj_novel = max(1, int(len(object_labels) * args.object_novel_ratio)) if object_labels else 0
    n_rel_novel = max(1, int(len(relation_labels) * args.relation_novel_ratio)) if relation_labels else 0
    object_novel, relation_novel = set(object_labels[:n_obj_novel]), set(relation_labels[:n_rel_novel])

    def has_object(rec, labels):
        return any(obj.get("label") in labels for obj in rec.get("objects", []))

    def has_relation(rec, labels):
        return any(rel.get("predicate") in labels for rel in rec.get("relations", []))

    def is_ss(rec):
        return not has_object(rec, object_novel) and not has_relation(rec, relation_novel)

    def is_ns(rec):
        return has_object(rec, object_novel) and not has_relation(rec, relation_novel)

    def is_sn(rec):
        return not has_object(rec, object_novel) and has_relation(rec, relation_novel)

    def is_nn(rec):
        return has_object(rec, object_novel) and has_relation(rec, relation_novel)

    out = Path(args.output_dir)
    for name, fn in [("ss", is_ss), ("ns", is_ns), ("sn", is_sn), ("nn", is_nn)]:
        write_jsonl(out / f"{name}.jsonl", [r for r in records if fn(r)])
    (out / "split_manifest.json").write_text(json.dumps({
        "object_novel": sorted(object_novel), "relation_novel": sorted(relation_novel),
        "object_counts": object_count, "relation_counts": relation_count,
        "seed": args.seed,
    }, indent=2, default=dict), encoding="utf-8")


if __name__ == "__main__":
    main()
