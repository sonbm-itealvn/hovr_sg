from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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


def record_image_id(record: dict, index: int) -> str:
    value = record.get("image_id", record.get("imageId"))
    return str(value) if value is not None else f"record_{index}"


def split_image_ids(image_ids: List[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    unique_ids = list(dict.fromkeys(image_ids))
    random.Random(seed).shuffle(unique_ids)
    if len(unique_ids) <= 1 or val_ratio == 0.0:
        return set(unique_ids), set()
    val_count = max(1, int(round(len(unique_ids) * val_ratio)))
    val_count = min(val_count, len(unique_ids) - 1)
    val_ids = set(unique_ids[:val_count])
    train_ids = set(unique_ids[val_count:])
    return train_ids, val_ids


def subset_by_image_ids(records: List[dict], selected_ids: set[str]) -> List[dict]:
    return [
        record for index, record in enumerate(records)
        if record_image_id(record, index) in selected_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build image-level train/val and open-vocabulary ss/ns/sn/nn splits"
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--object-novel-ratio", type=float, default=0.2)
    parser.add_argument("--relation-novel-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation fraction of the selected train/val image pool")
    parser.add_argument("--train-val-source", choices=["ss", "all"], default="ss",
                        help="Use strict seen-seen records or all records for train/val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 <= args.object_novel_ratio <= 1.0:
        raise ValueError("--object-novel-ratio must be in [0, 1]")
    if not 0.0 <= args.relation_novel_ratio <= 1.0:
        raise ValueError("--relation-novel-ratio must be in [0, 1]")

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

    def has_object(record, labels):
        return any(obj.get("label") in labels for obj in record.get("objects", []))

    def has_relation(record, labels):
        return any(rel.get("predicate") in labels for rel in record.get("relations", []))

    def is_ss(record):
        return not has_object(record, object_novel) and not has_relation(record, relation_novel)

    def is_ns(record):
        return has_object(record, object_novel) and not has_relation(record, relation_novel)

    def is_sn(record):
        return not has_object(record, object_novel) and has_relation(record, relation_novel)

    def is_nn(record):
        return has_object(record, object_novel) and has_relation(record, relation_novel)

    split_functions = {"ss": is_ss, "ns": is_ns, "sn": is_sn, "nn": is_nn}
    split_records = {name: [record for record in records if fn(record)] for name, fn in split_functions.items()}
    train_val_pool = split_records[args.train_val_source] if args.train_val_source == "ss" else records
    if not train_val_pool:
        raise ValueError(
            "The selected train/val pool is empty. Reduce novelty ratios or use "
            "--train-val-source all for a non-strict pilot split."
        )
    train_val_image_ids = [
        record_image_id(record, index) for index, record in enumerate(train_val_pool)
    ]
    train_ids, val_ids = split_image_ids(train_val_image_ids, args.val_ratio, args.seed)
    train_records = subset_by_image_ids(train_val_pool, train_ids)
    val_records = subset_by_image_ids(train_val_pool, val_ids)

    out = Path(args.output_dir)
    for name, split in split_records.items():
        write_jsonl(out / f"{name}.jsonl", split)
    write_jsonl(out / "train.jsonl", train_records)
    write_jsonl(out / "val.jsonl", val_records)
    manifest = {
        "input": [str(path) for path in args.input],
        "ontology": str(args.ontology),
        "seed": args.seed,
        "object_novel_ratio": args.object_novel_ratio,
        "relation_novel_ratio": args.relation_novel_ratio,
        "val_ratio": args.val_ratio,
        "train_val_source": args.train_val_source,
        "object_novel": sorted(object_novel),
        "relation_novel": sorted(relation_novel),
        "object_counts": dict(object_count),
        "relation_counts": dict(relation_count),
        "record_counts": {name: len(split) for name, split in split_records.items()},
        "image_counts": {
            name: len({record_image_id(record, index) for index, record in enumerate(split)})
            for name, split in split_records.items()
        },
        "train_val_pool_images": len(set(train_val_image_ids)),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "train_val_image_overlap": len(train_ids & val_ids),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
