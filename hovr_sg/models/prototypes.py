from __future__ import annotations

import hashlib
from typing import Iterable, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def deterministic_text_embeddings(labels: Iterable[str], dim: int = 512) -> Tensor:
    """Create stable pseudo-text vectors for smoke tests only.

    This is not a language model. Replace it with CLIP/SigLIP text embeddings for
    real open-vocabulary experiments.
    """
    rows: List[Tensor] = []
    for label in labels:
        digest = hashlib.sha256(label.strip().lower().encode("utf-8")).digest()
        raw = bytearray()
        while len(raw) < dim * 4:
            raw.extend(hashlib.sha256(bytes(raw[-32:]) + digest).digest())
        vals = torch.tensor(list(raw[: dim * 4]), dtype=torch.float32).view(-1, 4).mean(-1)
        vals = (vals - vals.mean()) / vals.std().clamp_min(1e-6)
        rows.append(vals)
    return F.normalize(torch.stack(rows), dim=-1)


class PrototypeBank(nn.Module):
    """Stores fixed or trainable text prototypes for three vocabularies."""

    def __init__(self, leaf: Tensor, groups: Tensor, relations: Tensor, trainable: bool = False):
        super().__init__()
        cls = nn.Parameter if trainable else lambda x: x
        self.leaf = cls(F.normalize(leaf.float(), dim=-1))
        self.groups = cls(F.normalize(groups.float(), dim=-1))
        self.relations = cls(F.normalize(relations.float(), dim=-1))

    def forward(self):
        return self.leaf, self.groups, self.relations
