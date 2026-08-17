from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ObjectAnnotation:
    id: int
    bbox: List[float]
    label: str
    group_labels: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    is_group: bool = False
    source: str = "unknown"


@dataclass
class RelationAnnotation:
    subject_id: int
    object_id: int
    predicate: str
    predicate_group: Optional[str] = None
    source: str = "unknown"


@dataclass
class SceneRecord:
    image_id: str
    image_path: str
    width: int
    height: int
    objects: List[ObjectAnnotation]
    relations: List[RelationAnnotation]
    annotation_scope: str = "unknown"
    negative_labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SceneRecord":
        return cls(
            image_id=str(raw["image_id"]),
            image_path=str(raw.get("image_path", "")),
            width=int(raw.get("width", 0)),
            height=int(raw.get("height", 0)),
            objects=[ObjectAnnotation(**obj) for obj in raw.get("objects", [])],
            relations=[RelationAnnotation(**rel) for rel in raw.get("relations", [])],
            annotation_scope=raw.get("annotation_scope", "unknown"),
            negative_labels=list(raw.get("negative_labels", [])),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "objects": [obj.__dict__ for obj in self.objects],
            "relations": [rel.__dict__ for rel in self.relations],
            "annotation_scope": self.annotation_scope,
            "negative_labels": self.negative_labels,
            "metadata": self.metadata,
        }
