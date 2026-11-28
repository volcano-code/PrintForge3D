# Cloud-first status — Phase 1.6

## Implemented in this tranche

- Dependency-free weighted sparse TSDF reference implementation.
- Dependency-free triangle mesh topology checks for degenerate faces, boundary edges, non-manifold edges, and combinatorial watertightness.
- Unit tests for both CPU reference modules.
- CPU development Dockerfile.
- Cloud-first development and CI policy documentation.
- GPU validation script that reports the environment and fails explicitly if CUDA is unavailable.

## Not yet implemented or validated

- No hosted GPU runner is configured.
- No foundation-model weights have been downloaded in CI.
- No MapAnything/VGGT/Déjà View inference has been executed by this tranche.
- The TSDF module is a sparse scalar reference accumulator, not a camera-ray integration or high-performance production fusion engine.
- Mesh topology checks do not detect geometric self-intersections and do not repair meshes.
- No DTU reconstruction or real-print experiment has been run by this tranche.

## Next steps

1. Run the new tests in GitHub Actions.
2. Add dedicated jobs for geometry and model contracts if the workflow grows.
3. Implement a real depth/camera-to-TSDF integration backend with a deterministic synthetic fixture.
4. Add mesh geometric validation and repair proposals.
5. Validate MapAnything adapter against a pinned upstream revision and a real checkpoint on a GPU.
