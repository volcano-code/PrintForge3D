# Development setup

## Supported baseline
- Linux x86_64 (primary target)
- Python 3.10+
- CMake 3.20+
- C++17 compiler (GCC 11+ or Clang 14+ recommended)
- COLMAP installed separately for reconstruction runs

The dependency-light Python tests and C++ core must work without CUDA, COLMAP, Open3D, or model weights.

## Python
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check src tests
printforge doctor
```

Optional vision dependencies:
```bash
python -m pip install -e ".[vision]"
```

## C++
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Optional Python bindings require pybind11:
```bash
python -m pip install pybind11
cmake -S . -B build-python -DPRINTFORGE_BUILD_PYTHON=ON \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build-python --parallel
```

## COLMAP sparse baseline
```bash
printforge colmap-run --images /path/to/images --workspace /path/to/runs/object-001
```

The runner executes feature extraction, exhaustive matching, and sparse mapping. It does not yet run dense MVS, mesh extraction, or printability checks. Inspect colmap.log and preserve all arguments with each experiment.
