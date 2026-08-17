"""Text prototype construction for open-vocabulary HOVR-SG heads."""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def deterministic_text_embeddings(labels: Iterable[str], dim: int = 512) -> Tensor:
    """Create stable pseudo-text vectors for smoke tests only.

    This is not a language model and must not be used as the text side of a real
    open-vocabulary experiment.  Use :class:`CLIPTextPrototypeEncoder` with the
    same CLIP checkpoint as the vision backbone instead.
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


class CLIPTextPrototypeEncoder:
    """Encode ontology labels with the text tower paired with a CLIP vision tower.

    The returned width is the checkpoint's ``projection_dim`` (512 for the
    standard OpenAI ViT-B/32 checkpoint), which is the required ``d_latent`` for
    the prototype heads.  Region/query features are projected into this space by
    HOVR-SG's learned object and relation projection layers.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        prompt_template: str = "a photo of a {label}",
        local_files_only: bool = False,
        device: torch.device | str = "cpu",
    ):
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise ImportError(
                "CLIPTextPrototypeEncoder requires transformers. "
                "Install the research dependencies with `pip install -e \".[vlm]\"`."
            ) from exc

        self.model_name = model_name
        self.prompt_template = prompt_template
        self.device = torch.device(device)
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.output_dim = int(getattr(
            self.model.config, "projection_dim", self.model.text_projection.out_features
        ))

    @torch.no_grad()
    def encode(self, labels: Sequence[str]) -> Tensor:
        if not labels:
            raise ValueError("At least one label is required to build text prototypes")
        texts = [self.prompt_template.format(label=label) for label in labels]
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**tokens)
        embeddings = outputs.text_embeds
        return F.normalize(embeddings.float(), dim=-1)


class PrototypeBank(nn.Module):
    """Stores fixed or trainable text prototypes for three vocabularies."""

    def __init__(self, leaf: Tensor, groups: Tensor, relations: Tensor, trainable: bool = False):
        super().__init__()
        values = {
            "leaf": F.normalize(leaf.float(), dim=-1),
            "groups": F.normalize(groups.float(), dim=-1),
            "relations": F.normalize(relations.float(), dim=-1),
        }
        for name, value in values.items():
            if trainable:
                setattr(self, name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)
        self.embedding_dim = int(values["leaf"].shape[-1])
        if any(value.shape[-1] != self.embedding_dim for value in values.values()):
            raise ValueError("Leaf, group and relation prototypes must share one embedding dimension")

    def forward(self):
        return self.leaf, self.groups, self.relations
