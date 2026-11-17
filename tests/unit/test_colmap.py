from pathlib import Path
import pytest
from printforge.reconstruction.colmap import run_colmap

def test_colmap_requires_existing_images_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_colmap(tmp_path / "missing", tmp_path / "out")

def test_colmap_requires_at_least_two_images(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="at least two"):
        run_colmap(images, tmp_path / "out")
