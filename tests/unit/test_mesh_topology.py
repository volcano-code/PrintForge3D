import pytest

from printforge.mesh.topology import analyze_triangle_mesh


def test_closed_tetrahedron_is_watertight():
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    report = analyze_triangle_mesh(vertices, faces)
    assert report.watertight
    assert report.boundary_edge_count == 0
    assert report.non_manifold_edge_count == 0


def test_open_triangle_has_three_boundary_edges():
    report = analyze_triangle_mesh(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        [(0, 1, 2)],
    )
    assert not report.watertight
    assert report.boundary_edge_count == 3


def test_duplicate_index_face_is_degenerate():
    report = analyze_triangle_mesh(
        [(0, 0, 0), (1, 0, 0)],
        [(0, 1, 1)],
    )
    assert report.degenerate_face_count == 1


def test_non_manifold_edge_is_reported():
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    faces = [(0, 1, 2), (1, 0, 3), (0, 1, 4)]
    report = analyze_triangle_mesh(vertices, faces)
    assert report.non_manifold_edge_count == 1
    assert not report.watertight


def test_out_of_range_index_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        analyze_triangle_mesh([(0, 0, 0)], [(0, 1, 2)])
