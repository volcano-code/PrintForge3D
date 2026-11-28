#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[dev]"
pytest -v
ruff check src tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
