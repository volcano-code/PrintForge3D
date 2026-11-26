def test_mapanything_adapter_import_does_not_load_torch_or_model():
    from printforge.foundation_models import MapAnythingAdapter

    assert callable(MapAnythingAdapter)
