from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image, ImageOps
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, Resize, ToTensor

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
        train: bool = False,
        augmentation: Optional[dict] = None,
    ):
        self.records = [json.loads(line) for line in Path(jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.ontology = ontology
        self.image_root = Path(image_root) if image_root else None
        self.train = bool(train)
        aug = augmentation or {}
        augmentation_enabled = self.train and bool(aug.get("enabled", True))
        self.hflip_prob = float(aug.get("hflip_prob", 0.5)) if augmentation_enabled else 0.0
        self.color_jitter = ColorJitter(
            brightness=float(aug.get("brightness", 0.2)),
            contrast=float(aug.get("contrast", 0.2)),
            saturation=float(aug.get("saturation", 0.2)),
            hue=float(aug.get("hue", 0.05)),
        ) if augmentation_enabled and bool(aug.get("color_jitter", True)) else None
        self.random_resized_crop = bool(aug.get("random_resized_crop", False)) and augmentation_enabled
        self.image_size = int(image_size)
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.transform = Compose([
            Resize((image_size, image_size)), ToTensor(),
            Normalize(self.image_mean, self.image_std),
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
        horizontal_flip = self.train and random.random() < self.hflip_prob
        if horizontal_flip:
            image = ImageOps.mirror(image)
        crop_params = None
        if self.random_resized_crop:
            crop_params = RandomResizedCrop.get_params(image, scale=(0.85, 1.0), ratio=(0.9, 1.1))
            top, left, crop_height, crop_width = crop_params
            image = TF.resized_crop(image, top, left, crop_height, crop_width,
                                    [self.image_size, self.image_size])
        if self.color_jitter is not None:
            image = self.color_jitter(image)
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
            x1_norm, y1_norm = x1 / max(record.width, 1), y1 / max(record.height, 1)
            x2_norm, y2_norm = x2 / max(record.width, 1), y2 / max(record.height, 1)
            if crop_params is not None:
                top, left, crop_height, crop_width = crop_params
                x1_norm = (x1 - left) / max(crop_width, 1)
                x2_norm = (x2 - left) / max(crop_width, 1)
                y1_norm = (y1 - top) / max(crop_height, 1)
                y2_norm = (y2 - top) / max(crop_height, 1)
            if horizontal_flip:
                x1_norm, x2_norm = 1.0 - x2_norm, 1.0 - x1_norm
            boxes.append([
                max(0.0, min(1.0, x1_norm)), max(0.0, min(1.0, y1_norm)),
                max(0.0, min(1.0, x2_norm)), max(0.0, min(1.0, y2_norm)),
            ])
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
