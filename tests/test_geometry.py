import numpy as np
import pytest
from synthetic import Pose, cylinder, plane, random_rotation

from fitpipeline import InsufficientGeometryError, prepare_segment
from fitpipeline.loader import build_segments


def test_weights_are_positive_and_sum_to_one():
    record, _ = cylinder()
    segment = prepare_segment(record)

    assert np.all(segment.vertex_weights > 0)
    assert np.all(segment.facet_weights > 0)
    assert segment.vertex_weights.sum() == pytest.approx(1.0)
    assert segment.facet_weights.sum() == pytest.approx(1.0)


def test_normalized_frame_is_centered_with_unit_rms_extent():
    record, _ = cylinder()
    segment = prepare_segment(record)

    centroid = segment.vertex_weights @ segment.vertices
    rms = np.sqrt(segment.vertex_weights @ (segment.vertices**2).sum(axis=1))
    np.testing.assert_allclose(centroid, 0, atol=0.1)  # vertex- vs area-weighted centroid
    assert 0.5 < rms < 1.5


def test_center_is_area_weighted_not_vertex_averaged():
    # A rectangle [0,4]x[-1,2] with grading toward u=0: the plain vertex mean is
    # pulled toward the dense side, the area centroid is exactly (2, 0.5, 0).
    record, _ = plane()
    segment = prepare_segment(record)

    assert not np.allclose(record.points.mean(axis=0), [2, 0.5, 0], atol=0.1)
    np.testing.assert_allclose(segment.normalization.center, [2, 0.5, 0], atol=1e-12)


def test_normalization_round_trips_points():
    record, _ = cylinder()
    normalization = prepare_segment(record).normalization

    restored = normalization.points_from_normalized(
        normalization.points_to_normalized(record.points)
    )
    np.testing.assert_allclose(restored, record.points, rtol=0, atol=1e-12)


def test_normalized_samples_ignore_translation_and_uniform_scale():
    base, _ = cylinder()
    moved, _ = cylinder(Pose(scale=250.0, translation=(1e3, -2e3, 5e2)))

    a, b = prepare_segment(base), prepare_segment(moved)

    np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-9)
    np.testing.assert_allclose(a.facet_normals, b.facet_normals, atol=1e-9)
    np.testing.assert_allclose(a.vertex_weights, b.vertex_weights, atol=1e-12)
    assert b.normalization.scale / a.normalization.scale == pytest.approx(250.0)


def test_normalized_samples_are_rotation_equivariant():
    rotation = random_rotation(3)
    base, _ = cylinder()
    rotated, _ = cylinder(Pose(rotation=rotation))

    a, b = prepare_segment(base), prepare_segment(rotated)

    np.testing.assert_allclose(a.vertices @ rotation.T, b.vertices, atol=1e-9)
    assert a.normalization.scale == pytest.approx(b.normalization.scale)


def test_far_from_origin_keeps_precision():
    base, _ = cylinder()
    far, _ = cylinder(Pose(translation=(1e7, -1e7, 1e7)))

    a, b = prepare_segment(base), prepare_segment(far)

    # Coordinates near 1e7 only carry ~1e-9 absolute precision in float64.
    np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-7)
    assert a.normalization.scale == pytest.approx(b.normalization.scale, rel=1e-8)


def test_degenerate_triangles_are_left_out_of_facet_samples():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0.0]])
    triangles = np.array([[0, 1, 2], [1, 3, 4]])  # second triangle is collinear
    record = build_segments("m", vertices, triangles, np.zeros(2, dtype=np.int64))[0]

    segment = prepare_segment(record)

    assert len(segment.facet_normals) == 1
    assert len(segment.vertices) == 3  # the degenerate triangle contributes no area


def test_zero_area_segment_is_rejected():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])
    record = build_segments("m", vertices, np.array([[0, 1, 2]]), np.zeros(1, dtype=np.int64))[0]

    with pytest.raises(InsufficientGeometryError, match="zero total area"):
        prepare_segment(record)


def test_non_finite_coordinates_are_rejected():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, np.nan, 0.0]])
    record = build_segments("m", vertices, np.array([[0, 1, 2]]), np.zeros(1, dtype=np.int64))[0]

    with pytest.raises(InsufficientGeometryError, match="non-finite"):
        prepare_segment(record)
