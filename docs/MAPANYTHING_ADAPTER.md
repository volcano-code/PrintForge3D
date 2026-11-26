# MapAnything adapter (first foundation-model integration)

## Status

The optional adapter is implemented at `src/printforge/foundation_models/mapanything_adapter.py` and normalizes MapAnything's documented per-view outputs into `ReconstructionBatch`.

**Important:** this is an adapter implementation, not yet a successful local model run. The checkpoint has not been downloaded and inference has not been executed on the RTX 5090 D.

## License

Use only the Apache checkpoint:

- Model: `facebook/map-anything-apache`
- Code: Apache-2.0
- Do not use the default `facebook/map-anything` checkpoint for commercial-facing work; it is CC-BY-NC 4.0.

The adapter defaults to the Apache checkpoint. Record the checkpoint revision and license metadata for every experiment.

## Environment isolation

The upstream README currently recommends Python 3.12 and does not pin PyTorch/CUDA. Keep the existing `printforge311` environment unchanged.

Suggested isolated environment:

```bash
conda create -n printforge-mapanything python=3.12 -y
conda activate printforge-mapanything

python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/map-anything.git ~/third_party/map-anything
python -m pip install -e ~/third_party/map-anything

# Install this project into the isolated environment.
python -m pip install -e ~/PrintForge3D
```

Do not install MapAnything's `.[all]` extra initially. Add only optional dependencies required by the selected model and benchmark.

## Adapter usage

```python
from pathlib import Path

from printforge.foundation_models import MapAnythingAdapter

adapter = MapAnythingAdapter(
    checkpoint="facebook/map-anything-apache",
    device="cuda",
    resolution=518,
    memory_efficient=True,
    minibatch_size=1,
)

batch = adapter.predict(
    [
        Path("/data/PrintForge3D_Data/DTU/scene/images/00000000.png"),
        Path("/data/PrintForge3D_Data/DTU/scene/images/00000001.png"),
    ],
    config={"norm_type": "dinov2"},
)

print(batch.backend, batch.length_unit, len(batch.views))
```

## Output contract

The adapter consumes the documented MapAnything per-view keys:

- `pts3d`: world-coordinate point map (required by this adapter).
- `depth_z` or `depth_along_ray`: camera-frame depth.
- `conf`: per-pixel confidence.
- `mask`: validity mask.
- `intrinsics`: recovered pinhole intrinsics.

The contract records point-map and depth coordinate frames separately. Metric output is marked as model-predicted but not independently ground-truth verified.

## RTX 5090 D validation protocol

Before declaring the adapter integrated, run a staged profile:

1. 2 views at 518 resolution.
2. 4 views at 518 resolution.
3. 8 views at 518 resolution.
4. 16 views at 518 resolution.
5. Repeat with memory-efficient inference and minibatch size 1.

Record checkpoint revision, cold-start time, warm latency, peak allocated/reserved VRAM, host RAM, image count, and output tensor shapes. Do not increase to 32+ views until 16-view stability is demonstrated.

## Known limitations

- The current adapter requires MapAnything's world-coordinate `pts3d` output; it deliberately refuses to silently reinterpret camera-frame points as world points.
- Camera pose matrices are not yet normalized into the common contract. This is a follow-up adapter task after the first inference run confirms the exact output schema in the pinned upstream revision.
- The current unit tests validate the shared contract and import isolation, not numerical MapAnything inference.
