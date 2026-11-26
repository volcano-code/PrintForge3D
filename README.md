# PrintForge3D

**An open-source, print-aware multi-view 3D reconstruction and mesh-quality pipeline.**

PrintForge3D is an eight-week engineering and research project targeting reliable geometry reconstruction from multi-view RGB images, geometry-preserving mesh repair, and 3D-printability analysis.

> **Project status:** W1 foundation plus Phase 1.5 audit in progress. The COLMAP sparse-SfM runner, shared geometry prediction contract, model/license registry, and optional MapAnything Apache adapter are implemented. MapAnything weights have not been downloaded and no foundation-model inference has yet been run on the RTX 5090 D. Custom fusion, mesh repair, printability analysis, and the web application remain planned.

## Goals

1. Establish a reproducible classical reconstruction baseline using COLMAP.
2. Compare classical multi-view stereo with a licensed, reproducible learned reconstruction path.
3. Research cross-view reliability estimation for uncertainty-aware geometric fusion.
4. Build native C++ geometry primitives and profile performance before adding CUDA kernels.
5. Validate mesh topology and printability without silently changing user geometry.
6. Deliver a usable application with reproducible evaluation and real-print evidence.

## Planned pipeline

```text
Multi-view RGB + camera metadata
              |
              v
     COLMAP SfM / MVS baseline
              |                     Learned reconstruction (optional)
              +-------------------------------+
                              |
                              v
               Cross-view reliability estimates
                              |
                              v
                 Uncertainty-aware fusion
                              |
                              v
                 Surface / mesh extraction
                              |
                              v
             Mesh validation and repair proposals
                              |
                              v
        Printability checks -> STL / 3MF export
```

The first implementation prioritizes a measurable COLMAP baseline. A neural rendering representation such as 3D Gaussian Splatting is not treated as a watertight solid model by itself.

## Repository layout

- `src/printforge/`: Python package, CLI, reconstruction, foundation-model contracts/adapters, and evaluation.
- `cpp/`: native C++ geometry core and optional Python bindings.
- `tests/`: unit, integration, and regression tests.
- `configs/`: experiment and pipeline configuration.
- `datasets/`: manifests and documentation only; large/raw data is not committed.
- `docs/`: architecture, data protocol, benchmarks, and decisions.
- `scripts/`: reproducible developer and baseline helpers.

## Quick start (Python core)

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
printforge doctor
```

The core package deliberately keeps COLMAP and Open3D optional. Install the relevant system tools and Python extras when running reconstruction:

```bash
# Install COLMAP using your Linux distribution or the official build instructions.
# Then install optional Python vision dependencies:
python -m pip install -e ".[vision]"
printforge colmap-run --help
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, [docs/BASELINE_PROTOCOL.md](docs/BASELINE_PROTOCOL.md) for the first benchmark protocol, [docs/PHASE_1_5_TECH_AUDIT.md](docs/PHASE_1_5_TECH_AUDIT.md) for model/license decisions, and [docs/MAPANYTHING_ADAPTER.md](docs/MAPANYTHING_ADAPTER.md) for isolated model setup.

## Research principles

- Keep train/validation/test splits at the object/scene level to prevent leakage.
- Record camera intrinsics, coordinate conventions, scale, masks, and preprocessing.
- Compare methods using the same input views and evaluation protocol.
- Report geometry metrics only when a valid reference geometry exists.
- Separate geometric accuracy, topology validity, printability, runtime, and memory.
- Never claim an improvement without a reproducible experiment.
- Preserve source meshes; repairs are explicit outputs with before/after reports.

## Planned milestones

| Milestone | Scope | Status |
|---|---|---|
| W1 | Project foundation and COLMAP sparse-SfM runner | Code foundation complete; real DTU run pending |
| W2 | Learned reconstruction adapter and fair comparison | MapAnything adapter code added; runtime validation pending |
| W3 | Cross-view reliability features and dataset | Planned |
| W4 | Reliability network and weighted fusion | Planned |
| W5 | Mesh validation, repair proposals, printability checks | Planned |
| W6 | C++/CUDA profiling and acceleration | Planned |
| W7 | API, worker, and interactive web viewer | Planned |
| W8 | Ablations, real-print validation, release | Planned |

## License

Project source code is licensed under the MIT License. Third-party dependencies, model weights, and datasets retain their own licenses and terms; see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

## Disclaimer

Printability checks are advisory. A successful geometric check does not guarantee a successful print; material, printer, slicer, orientation, support strategy, and process settings also matter.
