import pytest
from printforge.evaluation.geometry import chamfer_distance, point_rmse

def test_chamfer_identical_clouds_is_zero():
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert chamfer_distance(points, points) == 0.0

def test_chamfer_is_symmetric():
    a, b = [(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    assert chamfer_distance(a, b) == chamfer_distance(b, a)

def test_chamfer_rejects_empty_cloud():
    with pytest.raises(ValueError):
        chamfer_distance([], [(0.0, 0.0, 0.0)])

def test_point_rmse():
    assert point_rmse([(0, 0, 0)], [(3, 4, 0)]) == 5.0

def test_point_rmse_requires_equal_lengths():
    with pytest.raises(ValueError):
        point_rmse([(0, 0, 0)], [])
