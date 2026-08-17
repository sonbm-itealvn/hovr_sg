from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class TinyImageEncoder(nn.Module):
    """Small fallback encoder; use a VLM/grounded backbone for real experiments."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), nn.GroupNorm(16, out_dim), nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.net(images)
        return features.flatten(2).transpose(1, 2)


class ExternalFeatureAdapter(nn.Module):
    """Adapter for externally computed VLM features shaped [B,S,D]."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("Expected [batch, sequence, feature_dim] visual features")
        return self.proj(features)
