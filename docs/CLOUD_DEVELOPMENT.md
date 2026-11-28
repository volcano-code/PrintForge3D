# Cloud-first development and evaluation

## Goals

Use cloud development and GitHub CI for all work that does not require a specific GPU or physical printer. Keep the user's RTX 5090 D environment for final CUDA, large-model inference/training, and real-print validation.

## Environment tiers

| Tier | Environment | Allowed work |
|---|---|---|
| A | ChatGPT Work / cloud development | Repository inspection, implementation, code review, docs, small CPU experiments |
| B | GitHub Actions hosted runners | Python tests, Ruff, CMake/CTest, contract tests, CPU geometry tests, Docker build |
| C | Optional GPU runner | CUDA build/tests and model inference if an explicitly provisioned GPU runner exists |
| D | User RTX 5090 D | Final target-hardware profiling, large checkpoints, training and CUDA validation |

GitHub-hosted standard runners must not be assumed to have NVIDIA GPUs or CUDA. A green CPU CI does not prove CUDA compatibility.

## Branch and pull request policy

- Work on a dedicated feature/research branch.
- Keep `main` stable.
- Run all required CI checks before merge.
- Do not merge merely because code was pushed.
- Record the exact commit and workflow run URL in each milestone report.

## Experiment artifact policy

Commit source code, small fixtures, configuration, manifests, and aggregate reports. Do not commit raw datasets, pretrained weights, large point clouds, or checkpoints. Store external artifact URIs, checksums, license metadata, and reproducibility instructions in manifests.

## Definition of done

A component is complete only when:
1. Unit and contract tests pass.
2. Ruff and C++ checks pass where applicable.
3. Inputs, coordinate conventions, units, and failure cases are documented.
4. Third-party code and weight licenses are recorded separately.
5. GPU-only claims are explicitly marked pending until run on a GPU runner or target workstation.
