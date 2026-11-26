"""Safe COLMAP sparse-SfM runner and text-model statistics parser."""

import shutil
import subprocess
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _run(command: list[str], log) -> None:
    log.write("$ " + " ".join(command) + "\\n")
    log.flush()
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"COLMAP command failed ({completed.returncode}); see the run log"
        )


def _parse_images_file(path: Path) -> int:
    """Count COLMAP image records by consuming header/observation line pairs."""
    lines = path.read_text(encoding="utf-8").splitlines()
    count = 0
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) >= 10:
            try:
                int(fields[0])
                for value in fields[1:8]:
                    float(value)
                int(fields[8])
            except ValueError:
                continue
            count += 1
            # Every image header is followed by one observations line, which
            # may be empty. Consume it so long observations are never misread.
            if index < len(lines):
                index += 1
    return count


def _parse_points_file(path: Path) -> tuple[int, float | None]:
    """Return point count and mean point reprojection error from points3D.txt."""
    errors: list[float] = []
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8:
            continue
        count += 1
        try:
            errors.append(float(fields[7]))
        except ValueError:
            continue
    return count, sum(errors) / len(errors) if errors else None


def analyze_text_model(model_dir: Path) -> dict[str, object]:
    """Parse a COLMAP TXT model directory (images.txt, points3D.txt)."""
    images_file = model_dir / "images.txt"
    points_file = model_dir / "points3D.txt"
    if not images_file.is_file() or not points_file.is_file():
        raise FileNotFoundError(
            f"expected images.txt and points3D.txt in COLMAP text model: {model_dir}"
        )
    point_count, mean_error = _parse_points_file(points_file)
    return {
        "registered_images": _parse_images_file(images_file),
        "sparse_points": point_count,
        "mean_track_reprojection_error_px": mean_error,
    }


def run_colmap(
    images: Path,
    workspace: Path,
    camera_model: str = "OPENCV",
    single_camera: bool = False,
    quality: str = "high",
) -> dict[str, object]:
    """Run feature extraction, exhaustive matching, mapping, and model analysis.

    This baseline intentionally stops at sparse SfM. Dense PatchMatch stereo
    is not invoked implicitly because the installed COLMAP build may lack CUDA.
    """
    images = images.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    if not images.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {images}")
    image_files = sorted(
        p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(image_files) < 2:
        raise ValueError("at least two supported image files are required")
    executable = shutil.which("colmap")
    if executable is None:
        raise FileNotFoundError("COLMAP executable not found on PATH")
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(
            f"workspace must be empty to avoid overwriting results: {workspace}"
        )

    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    text_dir = workspace / "sparse_text"
    workspace.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            executable,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.single_camera",
            "1" if single_camera else "0",
        ],
        [executable, "exhaustive_matcher", "--database_path", str(database)],
        [
            executable,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse_dir),
        ],
    ]
    log_path = workspace / "colmap.log"
    with log_path.open("w", encoding="utf-8") as log:
        for command in commands:
            _run(command, log)

        models = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
        if not models:
            raise RuntimeError(
                "COLMAP mapper produced no sparse model; inspect colmap.log"
            )

        model_stats: list[tuple[int, Path]] = []
        for index, model in enumerate(models):
            candidate_text = workspace / f"sparse_text_candidate_{index}"
            candidate_text.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    executable,
                    "model_converter",
                    "--input_path",
                    str(model),
                    "--output_path",
                    str(candidate_text),
                    "--output_type",
                    "TXT",
                ],
                log,
            )
            stats = analyze_text_model(candidate_text)
            model_stats.append((stats["registered_images"], model))

        best_model = max(model_stats, key=lambda item: item[0])[1]
        best_index = models.index(best_model)
        best_text = workspace / f"sparse_text_candidate_{best_index}"
        text_dir.mkdir(parents=True, exist_ok=True)
        for file in best_text.iterdir():
            shutil.copy2(file, text_dir / file.name)
        final_stats = analyze_text_model(text_dir)

    return {
        "backend": "COLMAP",
        "stage": "sparse_sfm",
        "input_images": len(image_files),
        "registered_images": final_stats["registered_images"],
        "registration_ratio": final_stats["registered_images"] / len(image_files),
        "sparse_points": final_stats["sparse_points"],
        "mean_track_reprojection_error_px": final_stats[
            "mean_track_reprojection_error_px"
        ],
        "database": str(database),
        "sparse_model": str(best_model),
        "text_model": str(text_dir),
        "log": str(log_path),
        "quality_profile": quality,
        "dense_mvs": "not_run",
        "status": "completed",
    }
