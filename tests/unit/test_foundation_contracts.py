import pytest

from printforge.foundation_models.contracts import (
    CameraIntrinsics,
    CoordinateFrame,
    LengthUnit,
    ReconstructionBatch,
    ViewPrediction,
)


def make_view(image_id="view-001"):
    return ViewPrediction(
        image_id=image_id,
        width=2,
        height=2,
        depth=[[1.0, 1.0], [1.0, 1.0]],
        confidence=[[0.9, 0.8], [0.7, 0.6]],
        pointmap=[
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        intrinsics=CameraIntrinsics(100.0, 100.0, 1.0, 1.0, 2, 2),
    )


def test_valid_reconstruction_batch():
    batch = ReconstructionBatch(
        backend="test-backend",
        coordinate_frame=CoordinateFrame.CAMERA,
        length_unit=LengthUnit.METER,
        views=(make_view(),),
    )
    assert batch.backend == "test-backend"
    assert batch.views[0].pointmap is not None


def test_depth_shape_must_match_image_dimensions():
    with pytest.raises(ValueError, match="depth must have shape"):
        ViewPrediction("view", 2, 2, depth=[[1.0, 1.0]])


def test_confidence_shape_must_match_depth():
    with pytest.raises(ValueError, match="confidence must have shape"):
        ViewPrediction(
            "view",
            2,
            2,
            depth=[[1.0, 1.0], [1.0, 1.0]],
            confidence=[[0.5]],
        )


def test_pointmap_must_have_xyz_channels():
    with pytest.raises(ValueError, match="pointmap must have shape"):
        ViewPrediction(
            "view",
            2,
            2,
            depth=[[1.0, 1.0], [1.0, 1.0]],
            pointmap=[[1.0, 2.0], [3.0, 4.0]],
        )


def test_image_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        ReconstructionBatch(
            backend="test",
            coordinate_frame=CoordinateFrame.WORLD,
            length_unit=LengthUnit.METER,
            views=(make_view("same"), make_view("same")),
        )


def test_unknown_units_cannot_claim_metric_scale():
    with pytest.raises(ValueError, match="unknown units"):
        ReconstructionBatch(
            backend="test",
            coordinate_frame=CoordinateFrame.WORLD,
            length_unit=LengthUnit.UNKNOWN,
            views=(make_view(),),
            metadata={"metric_scale_verified": True},
        )
