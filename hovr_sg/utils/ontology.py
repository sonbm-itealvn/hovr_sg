from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class LabelInfo:
    id: str
    name: str
    parents: List[str]
    aliases: List[str]
    siblings: List[str]


class Ontology:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.version = raw.get("version", "unknown")
        self.object_groups = self._read_labels(raw.get("object_groups", []))
        self.object_leaves = self._read_labels(raw.get("object_leaves", []))
        self.predicate_groups = self._read_labels(raw.get("predicate_groups", []))
        self.predicates = self._read_labels(raw.get("predicates", []))
        self._build_maps()

    @staticmethod
    def _read_labels(items: Iterable[dict]) -> Dict[str, LabelInfo]:
        return {
            item["id"]: LabelInfo(
                id=item["id"],
                name=item.get("name", item["id"]),
                parents=list(item.get("parents", [])),
                aliases=list(item.get("aliases", [])),
                siblings=list(item.get("siblings", [])),
            )
            for item in items
        }

    def _build_maps(self) -> None:
        self.leaf_to_idx = {k: i for i, k in enumerate(self.object_leaves)}
        self.group_to_idx = {k: i for i, k in enumerate(self.object_groups)}
        self.predicate_to_idx = {k: i for i, k in enumerate(self.predicates)}
        self.predicate_group_to_idx = {k: i for i, k in enumerate(self.predicate_groups)}
        self.leaf_to_groups: Dict[int, List[int]] = {
            self.leaf_to_idx[leaf]: [self.group_to_idx[parent] for parent in info.parents if parent in self.group_to_idx]
            for leaf, info in self.object_leaves.items()
        }
        self.alias_to_leaf: Dict[str, str] = {}
        for key, info in self.object_leaves.items():
            for alias in [key, info.name, *info.aliases]:
                self.alias_to_leaf[self._norm(alias)] = key
        self.alias_to_predicate: Dict[str, str] = {}
        for key, info in self.predicates.items():
            for alias in [key, info.name, *info.aliases]:
                self.alias_to_predicate[self._norm(alias)] = key

    @staticmethod
    def _norm(value: str) -> str:
        return " ".join(value.lower().replace("_", " ").split())

    def canonical_leaf(self, label: str) -> Optional[str]:
        return self.alias_to_leaf.get(self._norm(label))

    def canonical_predicate(self, label: str) -> Optional[str]:
        return self.alias_to_predicate.get(self._norm(label))

    def parent_groups(self, leaf: str) -> List[str]:
        info = self.object_leaves.get(leaf)
        return [] if info is None else list(info.parents)

    def predicate_groups_for(self, predicate: str) -> List[str]:
        info = self.predicates.get(predicate)
        return [] if info is None else list(info.parents)

    def leaf_names(self) -> List[str]:
        return list(self.object_leaves)

    def group_names(self) -> List[str]:
        return list(self.object_groups)

    def predicate_names(self) -> List[str]:
        return list(self.predicates)

    def group_index(self, name: str) -> int:
        return self.group_to_idx[name]

    def leaf_index(self, name: str) -> int:
        return self.leaf_to_idx[name]

    def predicate_index(self, name: str) -> int:
        return self.predicate_to_idx[name]

    def summary(self) -> dict:
        return {
            "version": self.version,
            "num_object_leaves": len(self.object_leaves),
            "num_object_groups": len(self.object_groups),
            "num_predicates": len(self.predicates),
            "num_predicate_groups": len(self.predicate_groups),
        }
