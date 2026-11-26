from pathlib import Path

import pytest

from printforge.reconstruction.colmap import analyze_text_model, run_colmap


def test_colmap_requires_existing_images_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_colmap(tmp_path / "missing", tmp_path / "out")


def test_colmap_requires_at_least_two_images(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="at least two"):
        run_colmap(images, tmp_path / "out")


def test_analyze_text_model_counts_registered_images_and_points(tmp_path: Path):
    images_lines = [
        "# header",
        "1 1 0 0 0 0 0 0 1 image_001.jpg",
        "0 0 -1",
        "2 1 0 0 0 1 0 0 1 image_002.jpg",
        "0 0 -1",
    ]
    points_lines = [
        "# header",
        "1 0 0 0 255 255 255 0.5 1 0 2 0",
        "2 1 0 0 255 0 0 1.5 1 1 2 1",
    ]
    (tmp_path / "images.txt").write_text("\n".join(images_lines), encoding="utf-8")
    (tmp_path / "points3D.txt").write_text("\n".join(points_lines), encoding="utf-8")

    stats = analyze_text_model(tmp_path)
    assert stats["registered_images"] == 2
    assert stats["sparse_points"] == 2
    assert stats["mean_track_reprojection_error_px"] == 1.0


def test_analyze_text_model_requires_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        analyze_text_model(tmp_path)
