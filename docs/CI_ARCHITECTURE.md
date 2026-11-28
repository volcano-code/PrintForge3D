# CI architecture

The current CI workflow validates Python packaging/tests/lint and the native C++ core. The cloud-first roadmap adds focused checks for geometry, model contracts, and reproducible CPU benchmarks.

## Required jobs

- Python: install dev extras, pytest, Ruff.
- C++: configure/build with CMake and run CTest.
- Geometry: exercise deterministic TSDF and mesh topology reference tests (currently included in Python test suite).
- Model contracts: test shapes, coordinate metadata, optional adapter imports without downloading weights.
- Container: build CPU development image when Docker build action is enabled.

## GPU tests

GPU checks are opt-in and must run only on a self-hosted or explicitly provisioned GPU runner. Standard GitHub-hosted runners do not establish CUDA correctness. GPU reports must include GPU model, driver, CUDA runtime, PyTorch version, peak VRAM, input resolution, view count, and warm/cold latency.

## Current status

The repository's existing CI runs the Python and C++ jobs. Geometry and model-contract tests are discovered by pytest. Dedicated GPU CI and cloud GPU runners are not configured.
