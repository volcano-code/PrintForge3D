# Baseline protocol (W1)

## Objective
Establish a reproducible classical sparse reconstruction baseline before introducing learned depth or custom fusion.

## Capture protocol
1. Capture at least 24 sharp images with substantial adjacent overlap (target 60–80%).
2. Keep exposure and focus stable; avoid motion blur.
3. Include multiple elevations and a full azimuth sweep.
4. Avoid featureless backgrounds and moving objects.
5. Record camera make/model, image dimensions, and calibration when available.
6. Include a scale reference when metric scale is required.

Split by object/scene, never by individual frame.

## W1 baseline
- COLMAP feature extraction.
- Exhaustive matching for small captures.
- Incremental mapper.
- Record registered image count, sparse point count, reprojection error, runtime, and peak memory when available.
- Save logs, commands, input manifest, and output model.

The current CLI reports sparse model directories and log location. Dense MVS and model-stat parsing are subsequent tasks.

## Comparison rules
All methods use identical input views and masks. Record camera coordinate conventions and scale alignment. Do not report Chamfer/F-score unless the reference is registered and sampled under a documented protocol.

## Required run manifest
Record run ID/time (UTC), object ID/split, image list/checksums, camera metadata, COLMAP version/arguments, git commit, hardware, runtime, peak memory, output paths, and validation status.
