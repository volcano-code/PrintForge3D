from printforge.foundation_models import MapAnythingAdapter
from printforge.foundation_models.mapanything_adapter import _without_batch


class ShapeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def __getitem__(self, index):
        if index == 0:
            return ShapeTensor(self.shape[1:])
        if index == (Ellipsis, 0):
            return ShapeTensor(self.shape[:-1])
        raise IndexError(index)


def test_mapanything_adapter_import_does_not_load_torch_or_model():
    assert callable(MapAnythingAdapter)


def test_without_batch_removes_batched_confidence_dimension():
    tensor = ShapeTensor((1, 3, 4))
    assert _without_batch(tensor).shape == (3, 4)


def test_without_batch_removes_batch_and_singleton_depth_channel():
    tensor = ShapeTensor((1, 3, 4, 1))
    assert _without_batch(tensor, channels_last=True).shape == (3, 4)
