from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from hovr_sg.data.schema import SceneRecord
from hovr_sg.utils.ontology import Ontology


class UnifiedSceneGraphDataset(Dataset):
    def __init__(
        self,
        jsonl: str | Path,
        ontology: Ontology,
        image_root: str | Path | None = None,
        image_size: int = 224,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.records = [json.loads(line) for line in Path(jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.ontology = ontology
        self.image_root = Path(image_root) if image_root else None
        self.transform = Compose([
            Resize((image_size, image_size)), ToTensor(),
            Normalize(list(image_mean), list(image_std)),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image(self, record: SceneRecord) -> Path:
        path = Path(record.image_path)
        if not path.is_absolute() and self.image_root:
            path = self.image_root / path
        return path

    def __getitem__(self, index: int) -> Dict:
        record = SceneRecord.from_dict(self.records[index])
        path = self._resolve_image(record)
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)
        boxes: List[List[float]] = []
        object_ids: List[int] = []
        leaf_indices: List[int] = []
        group_indices: List[List[int]] = []
        for obj in record.objects:
            label = self.ontology.canonical_leaf(obj.label)
            if label is None:
                continue
            x1, y1, x2, y2 = obj.bbox
            boxes.append([x1 / max(record.width, 1), y1 / max(record.height, 1),
                          x2 / max(record.width, 1), y2 / max(record.height, 1)])
            object_ids.append(int(obj.id))
            leaf_indices.append(self.ontology.leaf_index(label))
            groups = obj.group_labels or self.ontology.parent_groups(label)
            group_indices.append([self.ontology.group_index(g) for g in groups if g in self.ontology.group_to_idx])
        relation_targets = []
        for rel in record.relations:
            pred = self.ontology.canonical_predicate(rel.predicate)
            if pred is not None:
                relation_targets.append({
                    "subject_id": rel.subject_id,
                    "object_id": rel.object_id,
                    "predicate_index": self.ontology.predicate_index(pred),
                })
        return {
            "image": image_tensor,
            "image_id": record.image_id,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "object_ids": object_ids,
            "leaf_indices": torch.tensor(leaf_indices, dtype=torch.long),
            "group_indices": group_indices,
            "relations": relation_targets,
            "annotation_scope": record.annotation_scope,
        }


def collate_scene_graph(batch: List[Dict]) -> Dict:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "samples": batch,
    }
