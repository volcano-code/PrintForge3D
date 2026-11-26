# Phase 1.5 — Recent 3D Reconstruction Technology Audit

**Audit date:** 2026-11-26  
**Repository baseline:** `volcano-code/PrintForge3D`, `main` at `1e9f3657d5136861010c78ce2024a11b7281d688`  
**Purpose:** decide what may be evaluated, integrated, or independently reimplemented for PrintForge3D 3.0. This is a source/code/license audit, not a claim that any model has run on the RTX 5090 D.

## Executive decision

1. Build one common prediction contract for all feed-forward reconstruction models.
2. First runtime candidate: Déjà View, but its released checkpoint is **non-commercial**. Keep it in a research-only profile unless a separate commercial license is obtained.
3. First efficiency candidate: SAF3R on a supported backbone. The repository README documents VGGT, Pi3, DA3, and StreamVGGT support; use a separate environment because it installs custom Triton kernels.
4. 3R-Adapter is a strong research reference, but its own `LICENSE_STATUS.md` explicitly says no open-source license is granted. Do not vendor, copy, or adapt its source code. Implement any related ideas independently from the paper only after legal review.
5. TopoSurfel is research/non-profit only under its repository license and has a PyTorch 2.5.1 + CUDA 12.1 tested stack with several compiled CUDA dependencies. It is not a default commercial product dependency.
6. ADELE is a high-value mesh-native paper direction, but the official project page currently does not link a public implementation. Treat it as paper-only research until code and license are available.
7. LSRM is an object-centric comparison candidate, not a first deployment target: its README reports H200 testing, less than 40 GB inference memory, PyTorch 2.4.0/CUDA 12.1 pins, and gated DINOv3 weights. A 32 GB RTX 5090 D cannot be assumed to run it without optimization.

## Candidate registry

| Candidate | Verified source / status | Role in PrintForge3D | Decision |
|---|---|---|---|
| Déjà View / DVLT | Official NVIDIA Research page and `nv-tlabs/dvlt`; inference code, weights, evaluation and training code are listed as released | Recurrent feed-forward multi-view geometry baseline; compare refinement-step count K | **Research-only runtime candidate** until weight license is resolved |
| 3R-Adapter | Official `PKU-OV3-LAB/3R-Adapter`; NeurIPS 2026 Spotlight; license status explicitly says no open-source license | Paper reference for relation retrieval, rewiring and consistency refinement | **Do not reuse source**; independent implementation only after review |
| SAF3R | Official `jndeng/SAF3R`; README lists VGGT, Pi3, DA3 and StreamVGGT; Triton kernels | Training-free sparse-attention efficiency baseline | **Isolated benchmark candidate**; license needs confirmation |
| TopoSurfel | Official `Fan-Treasure/TopoSurfel`; code license is educational/research/non-profit only | Gaussian-surfel / proxy-mesh research comparison | **Research-only; not commercial dependency** |
| ADELE | Official ETH Zürich project page; SIGGRAPH Asia 2026 Conference Track; no public code link on project page | Mesh-native reconstruction research direction | **Paper-only until code/license appear** |
| LSRM | Official Meta repository; object-centric feed-forward reconstruction | Object-centric sparse-view comparison | **Defer**; H200 and <40 GB report exceed/strain 32 GB target; gated DINOv3 weights |
| AMB3R | CVPR 2026 paper; sparse compact volumetric backend and metric-scale feed-forward reconstruction | Metric-scale geometry baseline | **Keep in model audit**; verify exact implementation, license and checkpoint separately |
| VGGT | Official Meta research baseline | Common geometry foundation baseline | **Keep**; verify checkpoint and code terms separately |
| MapAnything | Official code Apache-2.0; two checkpoints have different licenses | Commercial-friendly feed-forward baseline | Use only `facebook/map-anything-apache` for commercial-friendly experiments |

## License and use restrictions

### Déjà View

- Repository code is primarily Apache-2.0.
- VGGT-derived files retain their own license.
- The `nvidia/dvlt` model weights are separately licensed under the NVIDIA License for non-commercial research/evaluation.
- Therefore, Apache licensing of the repository does **not** make the checkpoint commercially usable.

### 3R-Adapter

The official `LICENSE_STATUS.md` states: “No open-source license is granted by this repository snapshot.” The original code is all-rights-reserved. Public visibility is not a license to reuse it.

### TopoSurfel

Its `LICENSE.md` permits educational, research and non-profit use only; modifications distributed as derivatives must be open-source and commercial use is prohibited without contacting the authors. This conflicts with a commercial-facing product dependency.

### SAF3R

The repository currently has no top-level LICENSE file in the audited snapshot. Treat code as all-rights-reserved until the authors publish an explicit license or grant permission. Paper ideas may be studied and independently implemented with proper citation and legal review.

### ADELE

The official project page identifies the SIGGRAPH Asia 2026 paper and method, but does not currently link a public code release. No source code or checkpoint should be assumed available.

### LSRM

The README reports H200 testing and inference requiring less than 40 GB of GPU memory. It also requires gated DINOv3 ViT-H weights. Do not schedule it as a required 5090 D model until memory profiling and access approval are completed.

### MapAnything

The code is Apache-2.0, but checkpoints differ:
- `facebook/map-anything`: CC-BY-NC 4.0.
- `facebook/map-anything-apache`: Apache-2.0.

Use the Apache checkpoint for commercial-friendly experiments and record the exact checkpoint revision.

## Architecture decisions

### Main path

`COLMAP / camera metadata → feed-forward model adapter → common geometry contract → reliability/fusion → mesh validation`

Do not chain all foundation models together. Evaluate each model as an interchangeable producer of camera, depth, point-map and confidence outputs.

### First comparison set

1. COLMAP sparse reconstruction (existing baseline).
2. Déjà View (research-only checkpoint).
3. VGGT.
4. AMB3R.
5. MapAnything Apache checkpoint.

### Efficiency experiment

Run SAF3R as a separate patch/benchmark on a supported backbone. Compare:
- reconstruction quality before/after sparse attention;
- peak VRAM;
- latency and throughput;
- kernel compilation time;
- supported input view counts.

### Mesh experiment

- TopoSurfel: research-only benchmark if license conditions are accepted.
- ADELE: reproduce from paper only if enough implementation detail exists; otherwise keep as a design reference.
- LSRM: defer until 32 GB inference feasibility is demonstrated.

## RTX 5090 D compatibility policy

The user's base environment is Python 3.11.18, PyTorch 2.9.1+cu128, CUDA runtime 12.8, Open3D 0.20.0, RTX 5090 D (32 GB). Do not downgrade this environment for third-party repositories.

Each third-party model must run in an isolated environment or container. Record:
- Python / PyTorch / CUDA / compiler versions;
- model and checkpoint revision;
- peak VRAM and host RAM;
- cold-start and warm latency;
- input view count and image resolution;
- output coordinate frame and scale;
- license and dataset provenance.

A model is not “integrated” until its adapter passes contract tests and an end-to-end inference test on the target machine.

## Next implementation tasks

1. Add a shared typed geometry prediction contract.
2. Add contract tests for depth, confidence, point-map shape and coordinate metadata.
3. Add a model registry with code/weight license fields.
4. Add isolated environment recipes for DVLT and SAF3R.
5. Download no large checkpoint until the user confirms available disk space and the license profile.
6. Run DTU SampleSet through COLMAP and one feed-forward model before training CVRN.

## Sources checked

- Déjà View project: https://research.nvidia.com/labs/dvl/projects/dvlt/
- Déjà View code: https://github.com/nv-tlabs/dvlt
- 3R-Adapter code and license status: https://github.com/PKU-OV3-LAB/3R-Adapter
- SAF3R code: https://github.com/jndeng/SAF3R
- TopoSurfel code and license: https://github.com/Fan-Treasure/TopoSurfel
- ADELE official project page: https://vlg-acme.inf.ethz.ch/publications/ADELE-Adaptive-Delaunay-Grids-for.html
- LSRM code: https://github.com/facebookresearch/Large-Sparse-Reconstruction-Model
- AMB3R paper: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_AMB3R_Accurate_Feed-forward_Metric-scale_3D_Reconstruction_with_Backend_CVPR_2026_paper.html
- MapAnything code and model license notes: https://github.com/facebookresearch/map-anything
