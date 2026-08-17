"""Statistics and ontology generation utilities for Visual Genome."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OBJECT_GROUP_RULES = [
    ("person", r"\b(man|woman|boy|girl|person|people|human|child|children|baby|kid)\b"),
    ("animal", r"\b(dog|cat|bird|horse|sheep|cow|animal|bear|elephant|zebra|giraffe|lion|tiger)\b"),
    ("vehicle", r"\b(car|bus|truck|vehicle|bike|bicycle|motorcycle|train|boat|airplane|plane)\b"),
    ("furniture", r"\b(chair|sofa|couch|table|desk|bed|bench|cabinet|furniture|shelf)\b"),
    ("drinkware_container", r"\b(cup|mug|bottle|glass|plate|bowl|container|can)\b"),
    ("food", r"\b(food|pizza|cake|bread|fruit|apple|banana|sandwich|meal)\b"),
    ("clothing", r"\b(shirt|dress|shoe|hat|jacket|pants|clothing|sock|tie)\b"),
    ("electronics", r"\b(phone|computer|laptop|screen|tv|television|camera|keyboard|mouse)\b"),
    ("building", r"\b(building|house|home|window|door|wall|roof|room|street|road)\b"),
    ("plant", r"\b(plant|tree|flower|grass|leaf|branch)\b"),
    ("sports", r"\b(ball|bat|racket|skateboard|sport|sports)\b"),
]

DEFAULT_PREDICATE_GROUP_RULES = [
    ("contact_action", r"\b(hold|holding|carry|carrying|wear|wearing|touch|touching|sit|sitting|ride|riding)\b"),
    ("containment", r"\b(in|inside|contain|contains|within|cover|covered)\b"),
    ("vertical_spatial", r"\b(on|under|above|below|over|hanging|standing)\b"),
    ("directional", r"\b(left|right|front|behind|back|facing|next|beside|near|far)\b"),
    ("possession", r"\b(has|have|with|owned|belong)\b"),
    ("visual_relation", r"\b(look|looking|see|visible|color|colou?r|made|shape)\b"),
]

DEFAULT_IRREGULARS = {
    "men": "man", "women": "woman", "people": "person", "children": "child",
    "mice": "mouse", "geese": "goose", "teeth": "tooth", "feet": "foot",
}


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records_by_image(raw: Any, key: str | None = None) -> dict[str, dict]:
    if isinstance(raw, dict):
        if key and key in raw and isinstance(raw[key], list):
            raw = raw[key]
        elif all(isinstance(value, dict) for value in raw.values()):
            return {str(image_id): value for image_id, value in raw.items()}
        else:
            raw = list(raw.values())
    output = {}
    for record in raw or []:
        image_id = record.get("image_id", record.get("id"))
        if image_id is not None:
            output[str(image_id)] = record
    return output


def normalize_label(label: str) -> str:
    text = str(label or "").strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def singularize(label: str) -> str:
    text = normalize_label(label)
    if text in DEFAULT_IRREGULARS:
        return DEFAULT_IRREGULARS[text]
    words = text.split()
    if not words:
        return text
    last = words[-1]
    if len(last) > 4 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 4 and last.endswith("ses"):
        last = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        last = last[:-1]
    words[-1] = last
    return " ".join(words)


def _policy_aliases(policy: dict) -> dict[str, str]:
    return {
        normalize_label(alias): singularize(canonical)
        for alias, canonical in policy.get("aliases", {}).items()
    }


def canonicalize(label: str, policy: dict, aliases: dict[str, str] | None = None) -> str:
    normalized = normalize_label(label)
    singular = singularize(normalized)
    aliases = aliases or _policy_aliases(policy)
    for candidate in (normalized, singular):
        if candidate in aliases:
            return aliases[candidate]
    return singular


def _rules(policy: dict, key: str, defaults: list[tuple[str, str]]) -> list[tuple[str, re.Pattern]]:
    configured = policy.get(key)
    raw_rules = configured if configured else [{"group": name, "pattern": pattern} for name, pattern in defaults]
    result = []
    for item in raw_rules:
        name = str(item.get("group", item.get("id", "other")))
        pattern = str(item.get("pattern", r"$^"))
        result.append((name, re.compile(pattern, re.IGNORECASE)))
    return result


def infer_group(label: str, policy: dict, rules: list[tuple[str, re.Pattern]], fallback: str) -> str:
    for group, pattern in rules:
        if pattern.search(label):
            return group
    return str(policy.get("fallback_object_group" if fallback == "other_object" else "fallback_predicate_group", fallback))


def _iter_object_labels(records: Iterable[dict]):
    for record in records:
        for obj in record.get("objects", []):
            labels = obj.get("names") or [obj.get("name", "")]
            for label in labels:
                if str(label).strip():
                    yield str(record.get("image_id", record.get("id", ""))), str(label)


def _iter_predicates(records: Iterable[dict]):
    for record in records:
        for relation in record.get("relationships", record.get("relations", [])):
            predicate = relation.get("predicate", relation.get("name", ""))
            if str(predicate).strip():
                yield str(record.get("image_id", record.get("id", ""))), str(predicate)


def _top_aliases(variants: Counter, canonical: str, max_aliases: int) -> list[str]:
    aliases = []
    for value, _ in variants.most_common():
        if normalize_label(value) != normalize_label(canonical) and value not in aliases:
            aliases.append(value)
        if len(aliases) >= max_aliases:
            break
    return aliases


def generate_vg_ontology(
    objects_path: str | Path,
    relationships_path: str | Path,
    policy: dict | None = None,
) -> tuple[dict, dict]:
    policy = policy or {}
    object_records = list(_records_by_image(_load_json(objects_path)).values())
    relation_records = list(_records_by_image(_load_json(relationships_path)).values())
    aliases = _policy_aliases(policy)
    object_counts = Counter()
    object_images = defaultdict(set)
    object_variants = defaultdict(Counter)
    for image_id, raw_label in _iter_object_labels(object_records):
        canonical = canonicalize(raw_label, policy, aliases)
        if canonical in {singularize(normalize_label(x)) for x in policy.get("ignore_labels", [])}:
            continue
        object_counts[canonical] += 1
        object_images[canonical].add(image_id)
        object_variants[canonical][raw_label] += 1

    predicate_counts = Counter()
    predicate_images = defaultdict(set)
    predicate_variants = defaultdict(Counter)
    for image_id, raw_predicate in _iter_predicates(relation_records):
        canonical = canonicalize(raw_predicate, policy, aliases)
        if canonical in {singularize(normalize_label(x)) for x in policy.get("ignore_predicates", [])}:
            continue
        predicate_counts[canonical] += 1
        predicate_images[canonical].add(image_id)
        predicate_variants[canonical][raw_predicate] += 1

    min_object = int(policy.get("min_object_count", 20))
    min_predicate = int(policy.get("min_predicate_count", 20))
    max_objects = int(policy.get("max_object_leaves", 0))
    max_predicates = int(policy.get("max_predicates", 0))
    object_candidates = [key for key, count in object_counts.most_common() if count >= min_object]
    predicate_candidates = [key for key, count in predicate_counts.most_common() if count >= min_predicate]
    if max_objects > 0:
        object_candidates = object_candidates[:max_objects]
    if max_predicates > 0:
        predicate_candidates = predicate_candidates[:max_predicates]

    object_rules = _rules(policy, "object_group_rules", DEFAULT_OBJECT_GROUP_RULES)
    predicate_rules = _rules(policy, "predicate_group_rules", DEFAULT_PREDICATE_GROUP_RULES)
    object_groups = {}
    object_parents = {}
    for label in object_candidates:
        group = infer_group(label, policy, object_rules, "other_object")
        object_groups[group] = {"id": group, "name": group.replace("_", " "), "parents": []}
        object_parents[label] = group
    predicate_groups = {}
    predicate_parents = {}
    for predicate in predicate_candidates:
        group = infer_group(predicate, policy, predicate_rules, "other_relation")
        predicate_groups[group] = {"id": group, "name": group.replace("_", " "), "parents": []}
        predicate_parents[predicate] = group

    siblings_by_group = defaultdict(list)
    for label, group in object_parents.items():
        siblings_by_group[group].append(label)
    max_aliases = int(policy.get("max_aliases_per_label", 12))
    object_leaves = []
    for label in object_candidates:
        siblings = [item for item in siblings_by_group[object_parents[label]] if item != label]
        object_leaves.append({
            "id": label.replace(" ", "_"),
            "name": label,
            "parents": [object_parents[label]],
            "aliases": _top_aliases(object_variants[label], label, max_aliases),
            "siblings": [item.replace(" ", "_") for item in siblings],
            "frequency": object_counts[label],
            "image_frequency": len(object_images[label]),
        })
    predicates = []
    symmetric = {normalize_label(item) for item in policy.get("symmetric_predicates", [])}
    for predicate in predicate_candidates:
        predicates.append({
            "id": predicate.replace(" ", "_"),
            "name": predicate,
            "parents": [predicate_parents[predicate]],
            "aliases": _top_aliases(predicate_variants[predicate], predicate, max_aliases),
            "symmetric": predicate in symmetric,
            "frequency": predicate_counts[predicate],
            "image_frequency": len(predicate_images[predicate]),
        })

    ontology = {
        "version": str(policy.get("version", "ontology_vg_v1")),
        "object_groups": list(object_groups.values()),
        "object_leaves": object_leaves,
        "predicate_groups": list(predicate_groups.values()),
        "predicates": predicates,
    }
    total_object = sum(object_counts.values())
    total_predicate = sum(predicate_counts.values())
    kept_object = sum(object_counts[label] for label in object_candidates)
    kept_predicate = sum(predicate_counts[label] for label in predicate_candidates)
    report = {
        "version": ontology["version"],
        "source": {"objects": str(objects_path), "relationships": str(relationships_path)},
        "object": {
            "raw_unique_labels": len(object_counts),
            "selected_unique_labels": len(object_candidates),
            "raw_instances": total_object,
            "selected_instances": kept_object,
            "coverage": kept_object / max(total_object, 1),
            "dropped_instances": total_object - kept_object,
            "top_labels": [{"label": label, "count": count} for label, count in object_counts.most_common(100)],
        },
        "predicate": {
            "raw_unique_labels": len(predicate_counts),
            "selected_unique_labels": len(predicate_candidates),
            "raw_instances": total_predicate,
            "selected_instances": kept_predicate,
            "coverage": kept_predicate / max(total_predicate, 1),
            "dropped_instances": total_predicate - kept_predicate,
            "top_labels": [{"predicate": label, "count": count} for label, count in predicate_counts.most_common(100)],
        },
        "policy": policy,
    }
    return ontology, report


def generate_ontology_from_statistics(
    object_counts: Counter,
    object_images: dict[str, set],
    object_variants: dict[str, Counter],
    predicate_counts: Counter,
    predicate_images: dict[str, set],
    predicate_variants: dict[str, Counter],
    policy: dict | None = None,
    source: str = "unknown",
) -> tuple[dict, dict]:
    """Build the same ontology schema from any dataset-specific statistics."""
    policy = policy or {}
    min_object = int(policy.get("min_object_count", 20))
    min_predicate = int(policy.get("min_predicate_count", 20))
    max_objects = int(policy.get("max_object_leaves", 0))
    max_predicates = int(policy.get("max_predicates", 0))
    ignored_objects = {singularize(normalize_label(x)) for x in policy.get("ignore_labels", [])}
    ignored_predicates = {singularize(normalize_label(x)) for x in policy.get("ignore_predicates", [])}
    object_candidates = [
        key for key, count in object_counts.most_common()
        if count >= min_object and key not in ignored_objects
    ]
    predicate_candidates = [
        key for key, count in predicate_counts.most_common()
        if count >= min_predicate and key not in ignored_predicates
    ]
    if max_objects > 0:
        object_candidates = object_candidates[:max_objects]
    if max_predicates > 0:
        predicate_candidates = predicate_candidates[:max_predicates]

    object_rules = _rules(policy, "object_group_rules", DEFAULT_OBJECT_GROUP_RULES)
    predicate_rules = _rules(policy, "predicate_group_rules", DEFAULT_PREDICATE_GROUP_RULES)
    object_groups = {}
    object_parents = {}
    for label in object_candidates:
        group = infer_group(label, policy, object_rules, "other_object")
        object_groups[group] = {"id": group, "name": group.replace("_", " "), "parents": []}
        object_parents[label] = group
    predicate_groups = {}
    predicate_parents = {}
    for predicate in predicate_candidates:
        group = infer_group(predicate, policy, predicate_rules, "other_relation")
        predicate_groups[group] = {"id": group, "name": group.replace("_", " "), "parents": []}
        predicate_parents[predicate] = group

    siblings_by_group = defaultdict(list)
    for label, group in object_parents.items():
        siblings_by_group[group].append(label)
    max_aliases = int(policy.get("max_aliases_per_label", 12))
    object_leaves = []
    for label in object_candidates:
        siblings = [item for item in siblings_by_group[object_parents[label]] if item != label]
        object_leaves.append({
            "id": label.replace(" ", "_"),
            "name": label,
            "parents": [object_parents[label]],
            "aliases": _top_aliases(object_variants[label], label, max_aliases),
            "siblings": [item.replace(" ", "_") for item in siblings],
            "frequency": object_counts[label],
            "image_frequency": len(object_images[label]),
        })
    symmetric = {normalize_label(item) for item in policy.get("symmetric_predicates", [])}
    predicates = []
    for predicate in predicate_candidates:
        predicates.append({
            "id": predicate.replace(" ", "_"),
            "name": predicate,
            "parents": [predicate_parents[predicate]],
            "aliases": _top_aliases(predicate_variants[predicate], predicate, max_aliases),
            "symmetric": predicate in symmetric,
            "frequency": predicate_counts[predicate],
            "image_frequency": len(predicate_images[predicate]),
        })
    ontology = {
        "version": str(policy.get("version", f"ontology_{source}_v1")),
        "object_groups": list(object_groups.values()),
        "object_leaves": object_leaves,
        "predicate_groups": list(predicate_groups.values()),
        "predicates": predicates,
    }
    total_object = sum(object_counts.values())
    total_predicate = sum(predicate_counts.values())
    kept_object = sum(object_counts[label] for label in object_candidates)
    kept_predicate = sum(predicate_counts[label] for label in predicate_candidates)
    report = {
        "version": ontology["version"],
        "source": source,
        "object": {
            "raw_unique_labels": len(object_counts),
            "selected_unique_labels": len(object_candidates),
            "raw_instances": total_object,
            "selected_instances": kept_object,
            "coverage": kept_object / max(total_object, 1),
            "dropped_instances": total_object - kept_object,
            "top_labels": [{"label": label, "count": count} for label, count in object_counts.most_common(100)],
        },
        "predicate": {
            "raw_unique_labels": len(predicate_counts),
            "selected_unique_labels": len(predicate_candidates),
            "raw_instances": total_predicate,
            "selected_instances": kept_predicate,
            "coverage": kept_predicate / max(total_predicate, 1),
            "dropped_instances": total_predicate - kept_predicate,
            "top_labels": [{"predicate": label, "count": count} for label, count in predicate_counts.most_common(100)],
        },
        "policy": policy,
    }
    return ontology, report
