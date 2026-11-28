"""Dependency-free triangle-mesh topology checks for CPU validation.

This module reports combinatorial properties only. It does not claim to detect
geometric self-intersections or to repair a mesh.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MeshTopologyReport:
    vertex_count: int
    face_count: int
    degenerate_face_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    watertight: bool


def analyze_triangle_mesh(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
) -> MeshTopologyReport:
    """Analyze triangle indices, degenerate faces and undirected edge incidence."""
    vertex_count = len(vertices)
    edge_incidence: Counter[tuple[int, int]] = Counter()
    degenerate = 0

    for face in faces:
        if len(face) != 3:
            raise ValueError("all faces must be triangles")
        indices = tuple(int(index) for index in face)
        if any(index < 0 or index >= vertex_count for index in indices):
            raise ValueError("face index is outside the vertex array")
        if len(set(indices)) != 3:
            degenerate += 1
            continue
        for start, end in (
            (indices[0], indices[1]),
            (indices[1], indices[2]),
            (indices[2], indices[0]),
        ):
            edge_incidence[tuple(sorted((start, end)))] += 1

    boundary = sum(count == 1 for count in edge_incidence.values())
    non_manifold = sum(count > 2 for count in edge_incidence.values())
    return MeshTopologyReport(
        vertex_count=vertex_count,
        face_count=len(faces),
        degenerate_face_count=degenerate,
        boundary_edge_count=boundary,
        non_manifold_edge_count=non_manifold,
        watertight=(
            bool(faces)
            and degenerate == 0
            and boundary == 0
            and non_manifold == 0
        ),
    )
