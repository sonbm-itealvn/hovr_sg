from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from hovr_sg.utils.ontology import Ontology


def xywh_to_xyxy(box: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def normalize_xyxy(box: Iterable[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [max(0.0, min(x1, width)), max(0.0, min(y1, height)),
            max(0.0, min(x2, width)), max(0.0, min(y2, height))]


def canonical_object(label: str, ontology: Ontology) -> Optional[Tuple[str, List[str]]]:
    leaf = ontology.canonical_leaf(label)
    if leaf is None:
        return None
    return leaf, ontology.parent_groups(leaf)


def canonical_relation(label: str, ontology: Ontology) -> Optional[Tuple[str, Optional[str]]]:
    pred = ontology.canonical_predicate(label)
    if pred is None:
        return None
    groups = ontology.predicate_groups_for(pred)
    return pred, groups[0] if groups else None


def relation_key(subject_id: int, object_id: int, predicate: str) -> Tuple[int, int, str]:
    return int(subject_id), int(object_id), predicate
