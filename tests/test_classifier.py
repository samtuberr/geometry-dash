from pathlib import Path

import numpy as np
import pytest
import synthetic
from synthetic import IDENTITY, Pose, grid_segment, random_rotation

from fitpipeline import classify_segment, load_all_meshes
from fitpipeline.classifier import max_dihedral_angle
from fitpipeline.loader import build_segments
from fitpipeline.primitives import PRIMITIVE_TYPES

DATA_DIR = Path(__file__).resolve().parents[1] / "assignment" / "data"

POSES = {
    "identity": IDENTITY,
    "everything": Pose(rotation=random_rotation(11), scale=0.01, translation=(-3.0, 2.0, 5.0)),
    "far_away": Pose(rotation=random_rotation(5), translation=(1e6, 2e6, -1e6)),
}
BUILDERS = {name: getattr(synthetic, name) for name in PRIMITIVE_TYPES}


def coarse_cylinder(sides=32, height=2.0, radius=2.0):
    """Two coaxial rings only: every vertex also lies exactly on a sphere."""
    return grid_segment(
        lambda t, z: (radius * np.cos(t), radius * np.sin(t), z),
        np.linspace(0.0, 2 * np.pi, sides, endpoint=False),
        np.array([-height / 2, height / 2]),
    )


def bumpy(pose=IDENTITY):
    """A freeform sheet: no supported primitive follows it."""
    return grid_segment(
        lambda u, v: (u, v, 0.35 * np.sin(2.3 * u) * np.cos(1.7 * v) + 0.1 * u * v),
        np.linspace(0, 3, 30),
        np.linspace(0, 3, 30),
        pose,
    )


@pytest.mark.parametrize("pose_name", POSES)
@pytest.mark.parametrize("primitive_type", PRIMITIVE_TYPES)
def test_known_primitive_is_classified_under_any_pose(primitive_type, pose_name):
    record, _ = BUILDERS[primitive_type](POSES[pose_name])

    result = classify_segment(record)

    assert result.kind == "primitive"
    assert result.primitive_type == primitive_type
    assert result.confidence > 0.9


@pytest.mark.parametrize("pose_name", POSES)
def test_freeform_surface_is_unresolved_under_any_pose(pose_name):
    result = classify_segment(bumpy(POSES[pose_name]))

    assert result.kind == "unresolved"
    assert result.primitive_type is None
    assert result.best is not None  # the closest attempt is kept for the report
    assert result.confidence > 0.7


def test_confidence_is_pose_invariant():
    reference = classify_segment(synthetic.cone(IDENTITY)[0])
    moved = classify_segment(synthetic.cone(POSES["everything"])[0])

    assert moved.confidence == pytest.approx(reference.confidence, abs=0.02)


def test_coarse_cylinder_is_not_mistaken_for_a_sphere():
    record = coarse_cylinder()

    result = classify_segment(record)

    assert result.primitive_type == "cylinder"
    sphere = next(c for c in result.candidates if c.primitive_type == "sphere")
    assert sphere.relative_rms < 1e-6  # the vertices alone cannot tell them apart
    assert sphere.normal_rms_radians > 1.5 * result.best.normal_rms_radians


def test_sphere_deformed_by_a_bump_is_rejected():
    record, truth = synthetic.sphere()
    points = record.points.copy()
    center = np.asarray(truth["center"])
    radial = points - center
    height = np.exp(-(np.linalg.norm(points - points[len(points) // 2], axis=1) / 0.6) ** 2)
    points += 0.25 * height[:, None] * radial / np.linalg.norm(radial, axis=1, keepdims=True)
    triangles = record.triangles
    deformed = build_segments("synthetic", points, triangles, np.zeros(len(triangles), dtype=np.int64))[0]

    assert classify_segment(deformed).kind == "unresolved"


def test_uneven_triangle_sizes_do_not_change_the_answer():
    # synthetic builders grade triangle sizes ~10x; a uniform grid must agree
    graded = classify_segment(synthetic.cylinder()[0])
    uniform = classify_segment(
        grid_segment(
            lambda t, z: (1.5 * np.cos(t), 1.5 * np.sin(t), z),
            np.linspace(0.2, 2.5, 20),
            np.linspace(0.0, 4.0, 5),
        )
    )

    assert graded.primitive_type == uniform.primitive_type == "cylinder"


def test_single_triangle_is_a_plane_without_full_confidence():
    record = build_segments(
        "synthetic",
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        np.array([[0, 1, 2]]),
        np.zeros(1, dtype=np.int64),
    )[0]

    result = classify_segment(record)

    assert result.primitive_type == "plane"
    assert result.confidence < classify_segment(synthetic.plane()[0]).confidence


def test_zero_area_segment_is_unresolved_not_an_exception():
    record = build_segments(
        "synthetic",
        np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        np.array([[0, 1, 2]]),
        np.zeros(1, dtype=np.int64),
    )[0]

    result = classify_segment(record)

    assert result.kind == "unresolved"
    assert "insufficient evidence" in result.reason


def test_curved_type_needs_more_vertices_than_parameters():
    # A 4-vertex cylinder strip (one quad): coplanar, so it is a plane, never a curved type.
    record = grid_segment(
        lambda t, z: (np.cos(t), np.sin(t), z), np.array([0.0, 0.4]), np.array([0.0, 1.0])
    )

    result = classify_segment(record)

    assert result.primitive_type == "plane"


def test_max_dihedral_angle_of_a_coarse_cylinder():
    assert max_dihedral_angle(coarse_cylinder(sides=32)) == pytest.approx(
        np.radians(360 / 32), rel=1e-6
    )


def test_max_dihedral_angle_of_a_flat_segment_is_zero():
    assert max_dihedral_angle(synthetic.plane()[0]) == pytest.approx(0.0, abs=1e-6)


def test_assignment_meshes_get_the_expected_labels():
    meshes = load_all_meshes(DATA_DIR)
    labels = {
        (mesh_id, segment_id): classify_segment(record).primitive_type
        for mesh_id, mesh in meshes.items()
        for segment_id, record in mesh.segments.items()
    }

    assert labels[("mesh_02", 0)] == "cylinder"
    assert labels[("mesh_03", 0)] == "cone"
    assert labels[("mesh_04", 0)] == "sphere"
    assert labels[("mesh_08", 0)] == "torus"
    assert {labels[("mesh_05", 0)], labels[("mesh_05", 1)]} == {"cylinder"}  # inner and outer wall
    assert labels[("mesh_09", 0)] is None  # deformed sphere
    assert labels[("mesh_10", 1)] is None  # freeform
    assert sum(label is None for label in labels.values()) == 2
    assert sum(label == "plane" for label in labels.values()) == 33
