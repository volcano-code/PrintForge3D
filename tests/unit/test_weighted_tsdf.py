import pytest

from printforge.fusion.weighted_tsdf import TSDFObservation, WeightedTSDF


def test_weighted_tsdf_computes_weighted_mean():
    volume = WeightedTSDF(truncation=2.0)
    volume.integrate(
        [
            TSDFObservation((0, 0, 0), -1.0, 1.0),
            TSDFObservation((0, 0, 0), 1.0, 3.0),
        ]
    )
    value = volume.get((0, 0, 0))
    assert value is not None
    assert value.signed_distance == pytest.approx(0.5)
    assert value.weight == pytest.approx(4.0)


def test_weighted_tsdf_clips_and_caps_weight():
    volume = WeightedTSDF(truncation=1.0, max_weight=2.0)
    volume.integrate([TSDFObservation((1, 2, 3), 5.0, 5.0)])
    value = volume.get((1, 2, 3))
    assert value is not None
    assert value.signed_distance == 1.0
    assert value.weight == 2.0


def test_weighted_tsdf_rejects_invalid_observations():
    volume = WeightedTSDF()
    with pytest.raises(ValueError):
        volume.integrate([TSDFObservation((0, 0, 0), float("nan"), 1.0)])
    with pytest.raises(ValueError):
        volume.integrate([TSDFObservation((0, 0, 0), 0.0, -1.0)])


def test_zero_weight_is_ignored():
    volume = WeightedTSDF()
    volume.integrate([TSDFObservation((0, 0, 0), 0.0, 0.0)])
    assert len(volume) == 0
