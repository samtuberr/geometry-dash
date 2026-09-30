import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synthetic
from synthetic import IDENTITY, Pose, random_rotation

from fitpipeline import load_mesh_segments, propose_hypotheses
from fitpipeline.loader import build_segments
from fitpipeline.primitives import PRIMITIVE_TYPES, surface_distance, surface_normal

DATA_DIR = Path(__file__).resolve().parents[1] / "assignment" / "data"

POSES = {
    "identity": IDENTITY,
    "rotated": Pose(rotation=random_rotation(7)),
    "scaled_moved": Pose(scale=40.0, translation=(300.0, -120.0, 75.0)),
    "everything": Pose(rotation=random_rotation(11), scale=0.01, translation=(-3.0, 2.0, 5.0)),
    "far_away": Pose(rotation=random_rotation(5), translation=(1e6, 2e6, -1e6)),
}
BUILDERS = {name: getattr(synthetic, name) for name in PRIMITIVE_TYPES}


def find(hypotheses, primitive_type):
    return next((h for h in hypotheses if h.primitive_type == primitive_type), None)


def axis_parallel(a, b):
    return abs(np.dot(a, b)) == pytest.approx(1.0, abs=1e-9)


def distance_from_line(point, line_point, line_direction):
    offset = np.asarray(point) - line_point
    return np.linalg.norm(offset - (offset @ line_direction) * line_direction)


@pytest.mark.parametrize("pose_name", POSES)
def test_plane_recovered(pose_name):
    pose = POSES[pose_name]
    record, truth = synthetic.plane(pose)
    guess = find(propose_hypotheses(record), "plane").parameters

    assert axis_parallel(guess["normal"], truth["normal"])
    on_plane_tolerance = 1e-9 * pose.scale + 1e-13 * np.abs(pose.translation).max()
    assert abs((guess["point"] - pose.point([0, 0, 0])) @ truth["normal"]) < on_plane_tolerance


@pytest.mark.parametrize("pose_name", POSES)
def test_sphere_recovered(pose_name):
    record, truth = synthetic.sphere(POSES[pose_name])
    guess = find(propose_hypotheses(record), "sphere").parameters

    assert guess["radius"] == pytest.approx(truth["radius"], rel=1e-8)
    np.testing.assert_allclose(
        guess["center"], truth["center"], rtol=0, atol=1e-8 * truth["radius"]
    )


@pytest.mark.parametrize("pose_name", POSES)
def test_cylinder_recovered(pose_name):
    record, truth = synthetic.cylinder(POSES[pose_name])
    guess = find(propose_hypotheses(record), "cylinder").parameters

    assert axis_parallel(guess["axis_direction"], truth["axis_direction"])
    assert guess["radius"] == pytest.approx(truth["radius"], rel=1e-8)
    assert distance_from_line(
        guess["axis_point"], truth["axis_point"], truth["axis_direction"]
    ) < 1e-8 * truth["radius"]


@pytest.mark.parametrize("pose_name", POSES)
def test_cone_recovered(pose_name):
    record, truth = synthetic.cone(POSES[pose_name])
    guess = find(propose_hypotheses(record), "cone").parameters

    np.testing.assert_allclose(guess["axis_direction"], truth["axis_direction"], atol=1e-8)
    assert guess["half_angle_radians"] == pytest.approx(truth["half_angle_radians"], abs=1e-8)
    scale = np.linalg.norm(truth["apex"] - POSES[pose_name].point([0, 0, 2]))
    np.testing.assert_allclose(guess["apex"], truth["apex"], rtol=0, atol=1e-7 * scale)


@pytest.mark.parametrize("pose_name", POSES)
def test_torus_recovered_within_tessellation_error(pose_name):
    # Facet normals of a tessellated torus are only approximately the surface
    # normals, so this is an initial guess (about 1e-4 here), not machine precision.
    record, truth = synthetic.torus(POSES[pose_name])
    guess = find(propose_hypotheses(record), "torus").parameters

    assert abs(np.dot(guess["axis_direction"], truth["axis_direction"])) > np.cos(1e-3)
    assert guess["major_radius"] == pytest.approx(truth["major_radius"], rel=1e-3)
    assert guess["minor_radius"] == pytest.approx(truth["minor_radius"], rel=1e-3)
    assert np.linalg.norm(guess["center"] - truth["center"]) < 1e-3 * truth["major_radius"]


@pytest.mark.parametrize("primitive_type", PRIMITIVE_TYPES)
def test_true_type_fits_and_residuals_do_not_depend_on_pose(primitive_type):
    reference, _ = BUILDERS[primitive_type](IDENTITY)
    posed, _ = BUILDERS[primitive_type](POSES["everything"])

    a = find(propose_hypotheses(reference), primitive_type)
    b = find(propose_hypotheses(posed), primitive_type)

    assert a.relative_rms < 1e-3  # torus is the loosest: chord normals bias its axis
    assert b.relative_rms == pytest.approx(a.relative_rms, abs=1e-7)
    assert b.normal_rms_radians == pytest.approx(a.normal_rms_radians, abs=1e-6)
    assert b.rms_distance == pytest.approx(
        a.rms_distance * POSES["everything"].scale, rel=1e-5, abs=1e-9
    )


def test_planar_segment_only_yields_a_plane_hypothesis():
    for pose in POSES.values():
        record, _ = synthetic.plane(pose)
        assert [h.primitive_type for h in propose_hypotheses(record)] == ["plane"]


@pytest.mark.parametrize("primitive_type", ["sphere", "cylinder", "cone", "torus"])
def test_plane_hypothesis_is_a_poor_fit_to_curved_surfaces(primitive_type):
    record, _ = BUILDERS[primitive_type]()
    plane_guess = find(propose_hypotheses(record), "plane")

    assert plane_guess.relative_rms > 0.05
    assert plane_guess.normal_rms_radians > 0.2


def test_wrong_curved_types_are_distinguishable_by_residuals():
    record, _ = synthetic.cylinder()
    hypotheses = propose_hypotheses(record)
    true = find(hypotheses, "cylinder")

    for other in hypotheses:
        if other.primitive_type != "cylinder":
            assert other.relative_rms > 100 * true.relative_rms or (
                other.normal_rms_radians > 5 * true.normal_rms_radians
            )


def test_freeform_surface_has_no_good_hypothesis():
    record = synthetic.grid_segment(
        lambda u, v: (u, v, 0.4 * np.sin(3 * u) * np.cos(2 * v)),
        np.linspace(0, 3, 40),
        np.linspace(0, 3, 40),
    )

    hypotheses = propose_hypotheses(record)

    assert hypotheses
    assert min(h.relative_rms for h in hypotheses) > 0.02


def test_single_triangle_gives_only_a_plane():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0.0]])
    record = build_segments("m", vertices, np.array([[0, 1, 2]]), np.zeros(1, dtype=np.int64))[0]

    hypotheses = propose_hypotheses(record)

    assert [h.primitive_type for h in hypotheses] == ["plane"]
    np.testing.assert_allclose(np.abs(hypotheses[0].parameters["normal"]), [0, 0, 1], atol=1e-12)


def test_plane_normal_follows_triangle_winding():
    record, _ = synthetic.plane()
    up = find(propose_hypotheses(record), "plane").parameters["normal"]
    flipped = replace(
        record, triangles=record.triangles[:, ::-1], triangle_normals=-record.triangle_normals
    )
    down = find(propose_hypotheses(flipped), "plane").parameters["normal"]

    np.testing.assert_allclose(up, -down, atol=1e-12)
    assert up @ record.triangle_normals[0] > 0.99


def test_results_are_bitwise_repeatable():
    record, _ = synthetic.torus(POSES["everything"])

    first, second = propose_hypotheses(record), propose_hypotheses(record)

    assert [h.primitive_type for h in first] == [h.primitive_type for h in second]
    for a, b in zip(first, second):
        assert a.relative_rms == b.relative_rms
        for key in a.parameters:
            assert np.array_equal(a.parameters[key], b.parameters[key])


def test_parameters_serialize_to_plain_json():
    record, _ = synthetic.cone()
    for hypothesis in propose_hypotheses(record):
        text = json.dumps(hypothesis.jsonable_parameters(), allow_nan=False)
        assert json.loads(text).keys() == hypothesis.parameters.keys()


def test_surface_functions_agree_on_a_known_point():
    center, axis = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0])
    on_surface = center + np.array([2.0, 0.0, 0.0])

    assert surface_distance("sphere", {"center": center, "radius": 2.0}, on_surface[None])[0] == 0
    cylinder = {"axis_point": center, "axis_direction": axis, "radius": 2.0}
    assert surface_distance("cylinder", cylinder, on_surface[None])[0] == 0
    torus = {"center": center, "axis_direction": axis, "major_radius": 1.0, "minor_radius": 1.0}
    assert surface_distance("torus", torus, on_surface[None])[0] == pytest.approx(0.0)
    np.testing.assert_allclose(surface_normal("torus", torus, on_surface[None]), [[1, 0, 0]])
    cone = {"apex": center, "axis_direction": axis, "half_angle_radians": np.pi / 4}
    np.testing.assert_allclose(
        surface_normal("cone", cone, np.array([[2.0, 0, 2.0]]) + center),
        [[-np.sqrt(0.5), 0, np.sqrt(0.5)]],
    )
    assert surface_distance("cone", cone, np.array([[0.0, 0, -1.0]]) + center)[0] == pytest.approx(1.0)


# --- real data --------------------------------------------------------------

MANIFEST = json.loads((DATA_DIR.parent / "manifest.json").read_text())
ALL_SEGMENTS = [
    (m["mesh_id"], int(segment_id))
    for m in MANIFEST["meshes"]
    for segment_id in m["segment_triangle_counts"]
]


@pytest.mark.parametrize("mesh_id,segment_id", ALL_SEGMENTS)
def test_every_supplied_segment_yields_finite_well_formed_hypotheses(mesh_id, segment_id):
    record = load_mesh_segments(DATA_DIR, mesh_id).segments[segment_id]

    hypotheses = propose_hypotheses(record)

    assert hypotheses, "every segment has at least a plane hypothesis"
    for hypothesis in hypotheses:
        assert np.isfinite(hypothesis.rms_distance) and np.isfinite(hypothesis.normal_rms_radians)
        for key, value in hypothesis.parameters.items():
            assert np.all(np.isfinite(value))
            if key in ("normal", "axis_direction"):
                assert np.linalg.norm(value) == pytest.approx(1.0)
            if key in ("radius", "major_radius", "minor_radius"):
                assert value > 0
        if "half_angle_radians" in hypothesis.parameters:
            assert 0 < hypothesis.parameters["half_angle_radians"] < np.pi / 2


def best(mesh_id, segment_id, primitive_type):
    record = load_mesh_segments(DATA_DIR, mesh_id).segments[segment_id]
    return find(propose_hypotheses(record), primitive_type)


def test_mesh_02_lateral_surface_is_a_radius_2_cylinder_along_z():
    guess = best("mesh_02", 0, "cylinder")

    assert guess.relative_rms < 1e-6
    assert guess.parameters["radius"] == pytest.approx(2.0, rel=1e-6)
    np.testing.assert_allclose(np.abs(guess.parameters["axis_direction"]), [0, 0, 1], atol=1e-6)


def test_coarse_cylinder_vertices_also_fit_a_sphere_but_normals_object():
    # Two coaxial rings of vertices lie on a sphere; only the facet normals tell.
    record = load_mesh_segments(DATA_DIR, "mesh_02").segments[0]
    hypotheses = propose_hypotheses(record)
    cyl, sph = find(hypotheses, "cylinder"), find(hypotheses, "sphere")

    assert sph.relative_rms < 1e-6
    assert sph.normal_rms_radians > 5 * cyl.normal_rms_radians


def test_mesh_03_side_is_a_cone_frustum_with_radii_2_and_1_over_height_4():
    guess = best("mesh_03", 0, "cone")

    assert guess.relative_rms < 1e-6
    assert guess.parameters["half_angle_radians"] == pytest.approx(np.arctan(0.25), abs=1e-6)
    np.testing.assert_allclose(guess.parameters["apex"], [0, 0, 6], atol=1e-6)
    assert guess.parameters["axis_direction"][2] < -0.999  # from the apex toward the frustum


def test_mesh_04_is_a_unit_sphere():
    guess = best("mesh_04", 0, "sphere")

    assert guess.relative_rms < 1e-6
    assert guess.parameters["radius"] == pytest.approx(1.0, rel=1e-6)
    np.testing.assert_allclose(guess.parameters["center"], 0, atol=1e-6)


def test_mesh_08_is_recovered_as_a_torus():
    guess = best("mesh_08", 0, "torus")

    assert guess.relative_rms < 1e-6
    assert guess.parameters["major_radius"] > guess.parameters["minor_radius"] > 0


@pytest.mark.parametrize("mesh_id,segment_id", [("mesh_09", 0), ("mesh_10", 1)])
def test_freeform_segments_have_no_good_primitive_hypothesis(mesh_id, segment_id):
    record = load_mesh_segments(DATA_DIR, mesh_id).segments[segment_id]

    hypotheses = propose_hypotheses(record)

    assert min(h.relative_rms for h in hypotheses) > 0.05
