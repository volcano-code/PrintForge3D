"""Shared, dependency-light contract for multi-view reconstruction backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class CoordinateFrame(str, Enum):
    """Coordinate frame used by predicted geometry."""

    WORLD = "world"
    CAMERA = "camera"


class LengthUnit(str, Enum):
    """Explicit length unit; UNKNOWN must not be silently treated as meters."""

    METER = "meter"
    MILLIMETER = "millimeter"
    UNKNOWN = "unknown"


def _shape(value: Any) -> tuple[int, ...]:
    """Read a tensor/array shape or infer it from nested Python sequences."""
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    child_shape = _shape(value[0])
    if any(_shape(child) != child_shape for child in value):
        raise ValueError("ragged arrays are not valid geometry tensors")
    return (len(value), *child_shape)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics and optional distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    model: str = "PINHOLE"
    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera image dimensions must be positive")
        if not self.model:
            raise ValueError("camera model must be specified")


@dataclass(frozen=True)
class ViewPrediction:
    """Per-view dense predictions in a declared coordinate convention."""

    image_id: str
    width: int
    height: int
    depth: Any
    confidence: Any | None = None
    pointmap: Any | None = None
    valid_mask: Any | None = None
    intrinsics: CameraIntrinsics | None = None
    world_from_camera: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("view dimensions must be positive")
        expected_2d = (self.height, self.width)
        if _shape(self.depth) != expected_2d:
            raise ValueError(f"depth must have shape {expected_2d}")
        if self.confidence is not None and _shape(self.confidence) != expected_2d:
            raise ValueError(f"confidence must have shape {expected_2d}")
        if self.valid_mask is not None and _shape(self.valid_mask) != expected_2d:
            raise ValueError(f"valid_mask must have shape {expected_2d}")
        if self.pointmap is not None and _shape(self.pointmap) != (*expected_2d, 3):
            raise ValueError(f"pointmap must have shape {(*expected_2d, 3)}")
        if self.intrinsics is not None:
            if (self.intrinsics.width, self.intrinsics.height) != (self.width, self.height):
                raise ValueError("intrinsics dimensions must match the prediction dimensions")
        if self.world_from_camera is not None:
            if len(self.world_from_camera) != 4 or any(
                len(row) != 4 for row in self.world_from_camera
            ):
                raise ValueError("world_from_camera must be a 4x4 homogeneous matrix")


@dataclass(frozen=True)
class ReconstructionBatch:
    """Model output normalized for fusion, evaluation, and downstream adapters."""

    backend: str
    coordinate_frame: CoordinateFrame
    length_unit: LengthUnit
    views: tuple[ViewPrediction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend name must be non-empty")
        if not self.views:
            raise ValueError("at least one view prediction is required")
        image_ids = [view.image_id for view in self.views]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("image_id values must be unique within a batch")
        if self.length_unit is LengthUnit.UNKNOWN:
            if self.metadata.get("metric_scale_verified", False):
                raise ValueError("unknown units cannot be marked as metric-scale verified")


class ReconstructionAdapter(Protocol):
    """Protocol implemented by every interchangeable feed-forward model adapter."""

    def predict(
        self,
        image_paths: Sequence[Path],
        *,
        config: Mapping[str, Any],
    ) -> ReconstructionBatch:
        """Predict per-view geometry and return the common reconstruction contract."""
        ...
