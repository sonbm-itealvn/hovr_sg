"""Class-balanced samplers for long-tail scene graph data.

Repeat Factor Sampling (Gupta et al., LVIS 2019) oversamples images
containing rare predicates so that the effective frequency distribution
is closer to uniform.  This prevents the relation decoder from
collapsing to majority predicates like 'on' or 'holding'.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler


class RepeatFactorSampler(Sampler[int]):
    """Repeat Factor Sampling for long-tail relation distributions.

    Each image *i* is repeated ``max(1, sqrt(t / f_c))`` times, where
    ``f_c`` is the category-level frequency of the rarest predicate
    present in image *i*, and ``t`` is a configurable threshold.

    Args:
        dataset: A dataset whose items contain ``"relations"`` entries
            with ``"predicate_index"`` keys.
        repeat_threshold: Frequency threshold *t*.  Categories with
            ``f_c < t`` get oversampled.  ``0.001`` is a good default
            for Visual Genome.
        seed: Random seed for epoch shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        repeat_threshold: float = 0.001,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.repeat_threshold = float(repeat_threshold)
        self.seed = int(seed)
        self._epoch = 0

        # Count per-category frequency across the dataset
        category_count: Counter[int] = Counter()
        image_categories: List[set] = []

        for idx in range(len(dataset)):
            # Access raw records to avoid heavy image I/O
            if hasattr(dataset, "records") and idx < len(dataset.records):
                raw = dataset.records[idx]
                cats: set[int] = set()
                for rel in raw.get("relations", []):
                    pred = rel.get("predicate_index")
                    if pred is not None and int(pred) >= 0:
                        cats.add(int(pred))
                        category_count[int(pred)] += 1
                image_categories.append(cats)
            else:
                image_categories.append(set())

        # Compute per-category frequency f_c = count_c / total
        total = max(sum(category_count.values()), 1)
        cat_freq: Dict[int, float] = {
            c: cnt / total for c, cnt in category_count.items()
        }

        # Compute per-image repeat factor
        self._repeat_factors: List[float] = []
        for cats in image_categories:
            if not cats:
                self._repeat_factors.append(1.0)
                continue
            # Use the rarest category in the image
            max_factor = max(
                math.sqrt(self.repeat_threshold / max(cat_freq.get(c, 1.0), 1e-10))
                for c in cats
            )
            self._repeat_factors.append(max(1.0, max_factor))

        # Build repeated index list
        self._int_part: List[int] = []
        self._frac_part: List[float] = []
        for idx, factor in enumerate(self._repeat_factors):
            int_count = int(factor)
            frac = factor - int_count
            self._int_part.extend([idx] * int_count)
            self._frac_part.append(frac)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic fractional sampling."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)

        # Deterministic fractional part: include index if random < frac
        indices = list(self._int_part)
        for idx, frac in enumerate(self._frac_part):
            if frac > 0 and torch.rand(1, generator=g).item() < frac:
                indices.append(idx)

        # Shuffle
        order = torch.randperm(len(indices), generator=g).tolist()
        return iter([indices[i] for i in order])

    def __len__(self) -> int:
        return len(self._int_part) + sum(1 for f in self._frac_part if f > 0)
