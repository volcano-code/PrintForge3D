"""Command-line interface for PrintForge3D."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from printforge import __version__
from printforge.reconstruction.colmap import run_colmap

def _doctor() -> int:
    checks = {"python": sys.version.split()[0],
              "colmap": shutil.which("colmap") or "not found (optional until reconstruction)"}
    for name in ("numpy", "open3d", "torch"):
        try:
            module = __import__(name)
            checks[name] = getattr(module, "__version__", "installed")
            if name == "torch":
                checks["cuda_available"] = str(module.cuda.is_available())
                if module.cuda.is_available():
                    checks["cuda_device_0"] = module.cuda.get_device_name(0)
        except ImportError:
            checks[name] = "not installed (optional)"
    print(json.dumps(checks, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="printforge", description="PrintForge3D toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="inspect optional local dependencies")
    colmap = sub.add_parser("colmap-run", help="run COLMAP sparse reconstruction")
    colmap.add_argument("--images", type=Path, required=True)
    colmap.add_argument("--workspace", type=Path, required=True)
    colmap.add_argument("--camera-model", default="OPENCV")
    colmap.add_argument("--single-camera", action="store_true")
    colmap.add_argument("--quality", choices=("low", "medium", "high", "extreme"), default="high")
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        return _doctor()
    if args.command == "colmap-run":
        try:
            result = run_colmap(args.images, args.workspace, args.camera_model,
                                args.single_camera, args.quality)
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
            parser.exit(2, f"printforge: error: {exc}\n")
        print(json.dumps(result, indent=2))
        return 0
    parser.error("unknown command")
    return 2
