#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
import platform
import torch

report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    report["gpu"] = torch.cuda.get_device_name(0)
    report["capability"] = torch.cuda.get_device_capability(0)
    report["total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
print(json.dumps(report, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; GPU validation cannot run")
PY
