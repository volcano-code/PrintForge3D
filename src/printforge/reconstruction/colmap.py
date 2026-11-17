"""Safe, explicit wrapper for the COLMAP command-line sparse SfM baseline."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Any

def run_colmap(images: Path, workspace: Path, camera_model: str = "OPENCV",
               single_camera: bool = False, quality: str = "high") -> dict[str, Any]:
    """Run feature extraction, exhaustive matching, and sparse mapping."""
    images = images.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    if not images.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {images}")
    image_files = [p for p in images.iterdir() if p.is_file()
                   and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
    if len(image_files) < 2:
        raise ValueError("at least two supported image files are required")
    executable = shutil.which("colmap")
    if executable is None:
        raise FileNotFoundError("COLMAP executable not found on PATH")
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"workspace must be empty to avoid overwriting results: {workspace}")
    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    workspace.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    prefix = [executable]
    commands = [
        prefix + ["feature_extractor", "--database_path", str(database), "--image_path", str(images),
                 "--ImageReader.camera_model", camera_model,
                 "--ImageReader.single_camera", "1" if single_camera else "0"],
        prefix + ["exhaustive_matcher", "--database_path", str(database)],
        prefix + ["mapper", "--database_path", str(database), "--image_path", str(images),
                 "--output_path", str(sparse_dir)],
    ]
    log_path = workspace / "colmap.log"
    with log_path.open("w", encoding="utf-8") as log:
        for command in commands:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"COLMAP command failed ({completed.returncode}); see {log_path}")
    models = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
    return {"backend": "COLMAP", "stage": "sparse_sfm", "image_count": len(image_files),
            "database": str(database), "sparse_models": [str(p) for p in models],
            "log": str(log_path), "quality_profile_reserved": quality, "status": "completed"}
