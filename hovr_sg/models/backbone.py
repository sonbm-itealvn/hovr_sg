"""Visual backbones used by HOVR-SG.

The tiny CNN is intentionally retained as an explicit smoke-test fallback.  The
research default is :class:`PretrainedCLIPVisionEncoder`, which exposes spatial
patch tokens from a pretrained CLIP vision tower.  Text prototypes are encoded
with the matching CLIP text tower in ``prototypes.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class TinyImageEncoder(nn.Module):
    """Small fallback encoder; use only for smoke tests, not open-vocabulary research."""

    visual_dim: int
    image_size: int = 224
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.visual_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, self.visual_dim, 3, stride=2, padding=1),
            nn.GroupNorm(16, self.visual_dim), nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.net(images)
        return features.flatten(2).transpose(1, 2)


class PretrainedCLIPVisionEncoder(nn.Module):
    """Return spatial patch tokens from a pretrained Hugging Face CLIP vision tower.

    ``visual_dim`` is read from the loaded checkpoint rather than hardcoded.  The
    class deliberately removes the CLS token because HOVR-SG needs spatial memory
    for query-based object and relation decoding.  The default freeze policy keeps
    the pretrained tower fixed; the last ``unfreeze_last_n_layers`` transformer
    blocks can be enabled for a staged fine-tuning experiment.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        trainable: bool = False,
        unfreeze_last_n_layers: int = 0,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from transformers import CLIPVisionModel
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise ImportError(
                "PretrainedCLIPVisionEncoder requires transformers. "
                "Install the research dependencies with `pip install -e \".[vlm]\"`."
            ) from exc

        self.model_name = model_name
        self.model = CLIPVisionModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        config = self.model.config
        self.visual_dim = int(config.hidden_size)
        self.image_size = int(config.image_size)
        self.patch_size = int(config.patch_size)
        self.image_mean = tuple(float(x) for x in getattr(
            config, "image_mean", (0.48145466, 0.4578275, 0.40821073)
        ))
        self.image_std = tuple(float(x) for x in getattr(
            config, "image_std", (0.26862954, 0.26130258, 0.27577711)
        ))
        self._configure_trainability(trainable, unfreeze_last_n_layers)

    def _configure_trainability(self, trainable: bool, unfreeze_last_n_layers: int) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = bool(trainable)
        if not trainable and unfreeze_last_n_layers > 0:
            layers = self.model.vision_model.encoder.layers
            count = min(int(unfreeze_last_n_layers), len(layers))
            for layer in layers[-count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.model.vision_model.post_layernorm.parameters():
                parameter.requires_grad = True

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.trainable_parameter_count == 0:
            # A frozen CLIP tower must not update LayerNorm/dropout behavior while
            # the detector is in training mode.
            self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images shaped [B, 3, H, W], got {tuple(images.shape)}")
        outputs = self.model(
            pixel_values=images,
            interpolate_pos_encoding=images.shape[-2:] != (self.image_size, self.image_size),
        )
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        if patch_tokens.shape[1] == 0:
            raise RuntimeError("CLIP vision backbone returned no spatial patch tokens")
        return patch_tokens


class ExternalFeatureAdapter(nn.Module):
    """Adapter for externally computed VLM features shaped ``[B, S, D]``."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("Expected [batch, sequence, feature_dim] visual features")
        return self.proj(features)


def build_image_encoder(model_cfg: dict) -> nn.Module:
    """Build the configured visual encoder and expose its resolved dimensions."""
    backbone = str(model_cfg.get("backbone", "clip")).lower()
    if backbone in {"clip", "pretrained_clip", "pretrained_vlm"}:
        return PretrainedCLIPVisionEncoder(
            model_name=str(model_cfg.get("backbone_name", "openai/clip-vit-base-patch32")),
            trainable=bool(model_cfg.get("train_backbone", False)),
            unfreeze_last_n_layers=int(model_cfg.get("unfreeze_last_n_layers", 0)),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
        )
    if backbone in {"tiny_cnn", "tiny"}:
        return TinyImageEncoder(int(model_cfg.get("visual_dim", 256)))
    raise ValueError(
        f"Unknown backbone={backbone!r}. Use 'clip' for pretrained research runs "
        "or explicitly 'tiny_cnn' for smoke tests."
    )
