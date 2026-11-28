"""Small deterministic weighted-TSDF reference used for tests and CPU baselines.

This implementation is intentionally dependency-free and is not a high-throughput
production fusion kernel. It provides a transparent numerical reference against
which future C++/CUDA implementations can be checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class TSDFObservation:
    """One normalized signed-distance observation for a voxel."""

    voxel: tuple[int, int, int]
    signed_distance: float
    weight: float


@dataclass(frozen=True)
class TSDFValue:
    """Accumulated signed distance and total confidence weight."""

    signed_distance: float
    weight: float


class WeightedTSDF:
    """Sparse weighted TSDF accumulator with deterministic update semantics."""

    def __init__(self, truncation: float = 1.0, max_weight: float = 100.0) -> None:
        if not isfinite(truncation) or truncation <= 0:
            raise ValueError("truncation must be finite and positive")
        if not isfinite(max_weight) or max_weight <= 0:
            raise ValueError("max_weight must be finite and positive")
        self.truncation = float(truncation)
        self.max_weight = float(max_weight)
        self._values: dict[tuple[int, int, int], TSDFValue] = {}

    def integrate(self, observations: Iterable[TSDFObservation]) -> None:
        """Integrate observations after clipping signed distance to truncation."""
        for observation in observations:
            distance = float(observation.signed_distance)
            weight = float(observation.weight)
            if not isfinite(distance):
                raise ValueError("signed distance must be finite")
            if not isfinite(weight) or weight < 0:
                raise ValueError("observation weight must be finite and non-negative")
            if len(observation.voxel) != 3:
                raise ValueError("voxel coordinate must contain exactly three integers")
            if weight == 0:
                continue

            clipped = max(-self.truncation, min(self.truncation, distance))
            old = self._values.get(observation.voxel, TSDFValue(0.0, 0.0))
            accepted_weight = min(weight, max(0.0, self.max_weight - old.weight))
            if accepted_weight == 0:
                continue
            total_weight = old.weight + accepted_weight
            fused = (
                old.signed_distance * old.weight + clipped * accepted_weight
            ) / total_weight
            self._values[observation.voxel] = TSDFValue(fused, total_weight)

    def get(self, voxel: tuple[int, int, int]) -> TSDFValue | None:
        """Return the accumulated value for a voxel, if observed."""
        return self._values.get(voxel)

    def items(self):
        """Return deterministic voxel/value pairs sorted lexicographically."""
        return tuple(sorted(self._values.items()))

    def __len__(self) -> int:
        return len(self._values)
