"""Optional MapAnything adapter using its documented Apache checkpoint path.

Install MapAnything in an isolated environment. The base PrintForge3D package
does not import MapAnything or PyTorch until this adapter is instantiated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from printforge.foundation_models.contracts import (
    CameraIntrinsics,
    CoordinateFrame,
    DepthConvention,
    LengthUnit,
    ReconstructionBatch,
    ViewPrediction,
)


def _without_batch(value: Any, *, channels_last: bool = False) -> Any:
    """Remove the single-scene batch dimension and optional singleton channel."""
    if not hasattr(value, "ndim") or value.ndim == 0:
        raise ValueError("model output must be a tensor with spatial dimensions")
    if value.ndim >= 3 and value.shape[0] == 1:
        value = value[0]
    if channels_last and value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    return value


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class MapAnythingAdapter:
    """Run the Apache-licensed MapAnything checkpoint and normalize its outputs."""

    def __init__(
        self,
        checkpoint: str = "facebook/map-anything-apache",
        device: str = "cuda",
        resolution: int = 518,
        patch_size: int = 14,
        memory_efficient: bool = True,
        minibatch_size: int = 1,
    ) -> None:
        try:
            import torch
            from mapanything.models import MapAnything
        except ImportError as exc:
            raise RuntimeError(
                "MapAnything is not installed in this environment. "
                "Install it in a dedicated environment using its official instructions."
            ) from exc

        self._torch = torch
        self.checkpoint = checkpoint
        self.device = device
        self.resolution = resolution
        self.patch_size = patch_size
        self.memory_efficient = memory_efficient
        self.minibatch_size = minibatch_size
        self.model = MapAnything.from_pretrained(checkpoint).to(device).eval()

    @staticmethod
    def _intrinsics(
        prediction: Mapping[str, Any], width: int, height: int
    ) -> CameraIntrinsics | None:
        matrix = prediction.get("intrinsics")
        if matrix is None:
            return None
        if matrix.ndim == 3 and matrix.shape[0] == 1:
            matrix = matrix[0]
        return CameraIntrinsics(
            fx=_scalar(matrix[0, 0]),
            fy=_scalar(matrix[1, 1]),
            cx=_scalar(matrix[0, 2]),
            cy=_scalar(matrix[1, 2]),
            width=width,
            height=height,
        )

    def predict(
        self,
        image_paths: Sequence[Path],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ReconstructionBatch:
        if not image_paths:
            raise ValueError("at least one image path is required")
        try:
            from mapanything.utils.image import load_images
        except ImportError as exc:
            raise RuntimeError(
                "MapAnything image utilities are unavailable; check the isolated install."
            ) from exc

        options = dict(config or {})
        resolution = int(options.get("resolution", self.resolution))
        norm_type = str(options.get("norm_type", "dinov2"))
        views = load_images(
            folder_or_list=[str(path) for path in image_paths],
            resolution_set=resolution,
            norm_type=norm_type,
            patch_size=self.patch_size,
        )

        with self._torch.no_grad():
            predictions = self.model.infer(
                views,
                memory_efficient_inference=self.memory_efficient,
                minibatch_size=self.minibatch_size,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                use_multiview_confidence=False,
            )

        if len(predictions) != len(image_paths):
            raise RuntimeError(
                "MapAnything returned a different number of predictions than input views"
            )

        normalized: list[ViewPrediction] = []
        for image_path, prediction in zip(image_paths, predictions, strict=True):
            pointmap = prediction.get("pts3d")
            if pointmap is None:
                raise RuntimeError(
                    "MapAnything output is missing pts3d in the world coordinate frame"
                )
            pointmap = _without_batch(pointmap)
            height, width = int(pointmap.shape[0]), int(pointmap.shape[1])

            depth = prediction.get("depth_z")
            depth_convention = DepthConvention.CAMERA_Z
            if depth is None:
                depth = prediction.get("depth_along_ray")
                depth_convention = DepthConvention.ALONG_RAY
            if depth is None:
                raise RuntimeError("MapAnything output is missing depth_z/depth_along_ray")
            depth = _without_batch(depth, channels_last=True)

            confidence = prediction.get("conf")
            if confidence is not None:
                confidence = _without_batch(confidence, channels_last=True)

            mask = prediction.get("mask")
            if mask is not None:
                mask = _without_batch(mask, channels_last=True)

            normalized.append(
                ViewPrediction(
                    image_id=str(image_path.resolve()),
                    width=width,
                    height=height,
                    depth=depth,
                    confidence=confidence,
                    pointmap=pointmap,
                    valid_mask=mask,
                    intrinsics=self._intrinsics(prediction, width, height),
                    depth_frame=CoordinateFrame.CAMERA,
                    pointmap_frame=CoordinateFrame.WORLD,
                    depth_convention=depth_convention,
                )
            )

        return ReconstructionBatch(
            backend="mapanything",
            coordinate_frame=CoordinateFrame.WORLD,
            length_unit=LengthUnit.METER,
            views=tuple(normalized),
            metadata={
                "checkpoint": self.checkpoint,
                "metric_scale_source": "MapAnything predicted metric geometry",
                "metric_scale_verified": False,
                "camera_convention": "OpenCV cam2world",
                "resolution": resolution,
                "memory_efficient_inference": self.memory_efficient,
                "minibatch_size": self.minibatch_size,
            },
        )
