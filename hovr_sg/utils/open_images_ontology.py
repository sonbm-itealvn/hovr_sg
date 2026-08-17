"""Open Images V6 ontology statistics."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from hovr_sg.utils.vg_ontology import (
    canonicalize,
    generate_ontology_from_statistics,
    normalize_label,
)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _first(row: dict, names: list[str], default: str = "") -> str:
    for name in names:
        if row.get(name) not in (None, ""):
            return str(row[name])
    return default


def _class_names(class_descriptions: str | Path | None) -> dict[str, str]:
    if not class_descriptions:
        return {}
    mapping = {}
    for row in read_csv(class_descriptions):
        key = _first(row, ["LabelName", "ClassID", "MID"])
        value = _first(row, ["DisplayName", "ClassName", "Name"], key)
        if key:
            mapping[key] = value
    return mapping


def generate_open_images_ontology(
    boxes_csv: str | Path,
    relationships_csv: str | Path | None = None,
    class_descriptions: str | Path | None = None,
    policy: dict | None = None,
) -> tuple[dict, dict]:
    policy = policy or {}
    aliases = {normalize_label(k): v for k, v in policy.get("aliases", {}).items()}
    names = _class_names(class_descriptions)
    object_counts = Counter()
    object_images = defaultdict(set)
    object_variants = defaultdict(Counter)
    for row in read_csv(boxes_csv):
        image_id = _first(row, ["ImageID", "image_id"])
        raw_label = names.get(_first(row, ["ClassName", "LabelName", "label"]), _first(row, ["ClassName", "LabelName", "label"]))
        if not raw_label:
            continue
        canonical = canonicalize(raw_label, policy, aliases)
        object_counts[canonical] += 1
        object_images[canonical].add(image_id)
        object_variants[canonical][raw_label] += 1

    predicate_counts = Counter()
    predicate_images = defaultdict(set)
    predicate_variants = defaultdict(Counter)
    if relationships_csv:
        for row in read_csv(relationships_csv):
            image_id = _first(row, ["ImageID", "image_id"])
            raw_predicate = _first(row, ["RelationshipLabel", "Predicate", "relation", "LabelName"])
            if not raw_predicate:
                continue
            canonical = canonicalize(raw_predicate, policy, aliases)
            predicate_counts[canonical] += 1
            predicate_images[canonical].add(image_id)
            predicate_variants[canonical][raw_predicate] += 1

    return generate_ontology_from_statistics(
        object_counts, object_images, object_variants,
        predicate_counts, predicate_images, predicate_variants,
        policy=policy, source="open_images_v6",
    )
