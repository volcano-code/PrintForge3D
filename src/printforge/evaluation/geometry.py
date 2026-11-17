"""Dependency-free geometry metrics for small point sets."""
from __future__ import annotations
import math
from collections.abc import Sequence
Point3 = tuple[float, float, float]

def chamfer_distance(points_a: Sequence[Point3], points_b: Sequence[Point3]) -> float:
    """Symmetric squared-L2 Chamfer distance (not normalized by scene scale)."""
    if not points_a or not points_b:
        raise ValueError("both point sets must be non-empty")
    def directed(source: Sequence[Point3], target: Sequence[Point3]) -> float:
        total = 0.0
        for point in source:
            total += min(sum((point[i] - q[i]) ** 2 for i in range(3)) for q in target)
        return total / len(source)
    return 0.5 * (directed(points_a, points_b) + directed(points_b, points_a))

def point_rmse(points_a: Sequence[Point3], points_b: Sequence[Point3]) -> float:
    """Paired point RMSE; requires correspondence and equal cardinality."""
    if not points_a or len(points_a) != len(points_b):
        raise ValueError("point sets must be non-empty and have equal lengths")
    squared = [sum((a[i] - b[i]) ** 2 for i in range(3))
               for a, b in zip(points_a, points_b, strict=True)]
    return math.sqrt(sum(squared) / len(squared))
