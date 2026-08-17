"""GQA scene-graph ontology statistics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hovr_sg.utils.vg_ontology import (
    canonicalize,
    generate_ontology_from_statistics,
    normalize_label,
)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _graphs(raw: Any) -> list[tuple[str, dict]]:
    if isinstance(raw, list):
        return [(str(item.get("imageId", item.get("image_id", item.get("id", "")))), item) for item in raw]
    if isinstance(raw, dict):
        return [(str(image_id), graph) for image_id, graph in raw.items()]
    raise ValueError("GQA scene graphs must be a JSON object or list")


def _object_items(raw_objects: Any):
    if isinstance(raw_objects, dict):
        return [(str(key), value) for key, value in raw_objects.items()]
    if isinstance(raw_objects, list):
        return [(str(index), value) for index, value in enumerate(raw_objects)]
    return []


def _relation_items(raw_relations: Any):
    if isinstance(raw_relations, dict):
        return [{"object": key, "name": value} for key, value in raw_relations.items()]
    return raw_relations if isinstance(raw_relations, list) else []


def generate_gqa_ontology(
    scene_graphs_json: str | Path,
    policy: dict | None = None,
) -> tuple[dict, dict]:
    policy = policy or {}
    aliases = {normalize_label(k): v for k, v in policy.get("aliases", {}).items()}
    object_counts = Counter()
    object_images = defaultdict(set)
    object_variants = defaultdict(Counter)
    predicate_counts = Counter()
    predicate_images = defaultdict(set)
    predicate_variants = defaultdict(Counter)

    for image_id, graph in _graphs(load_json(scene_graphs_json)):
        raw_objects = graph.get("objects", graph.get("nodes", {}))
        for _, raw_object in _object_items(raw_objects):
            raw_label = raw_object.get("name") or raw_object.get("label") or raw_object.get("class", "")
            if isinstance(raw_label, list):
                raw_label = raw_label[0] if raw_label else ""
            if not str(raw_label).strip():
                continue
            canonical = canonicalize(raw_label, policy, aliases)
            object_counts[canonical] += 1
            object_images[canonical].add(image_id)
            object_variants[canonical][str(raw_label)] += 1
            relations = raw_object.get("relations", raw_object.get("relationships", []))
            for relation in _relation_items(relations):
                raw_predicate = relation.get("name", relation.get("predicate", relation.get("relation", "")))
                if not str(raw_predicate).strip():
                    continue
                predicate = canonicalize(raw_predicate, policy, aliases)
                predicate_counts[predicate] += 1
                predicate_images[predicate].add(image_id)
                predicate_variants[predicate][str(raw_predicate)] += 1

    return generate_ontology_from_statistics(
        object_counts, object_images, object_variants,
        predicate_counts, predicate_images, predicate_variants,
        policy=policy, source="gqa",
    )
