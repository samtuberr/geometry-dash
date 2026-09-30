import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synthetic
from synthetic import IDENTITY, Pose, random_rotation

from fitpipeline import load_mesh_segments, propose_hypotheses, refine_hypothesis, refine_segment
from fitpipeline.loader import build_segments
from fitpipeline.primitives import (
    PRIMITIVE_TYPES,
    signed_surface_distance,
    surface_distance,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "assignment" / "data"

POSES = {
    "identity": IDENTITY,
    "everything": Pose(rotation=random_rotation(11), scale=0.01, translation=(-3.0, 2.0, 5.0)),
    "far_away": Pose(rotation=random_rotation(5), translation=(1e6, 2e6, -1e6)),
}
BUILDERS = {name: getattr(synthetic, name) for name in PRIMITIVE_TYPES}


def seed_for(record, primitive_type):
    return next(h for h in propose_hypotheses(record) if h.primitive_type == primitive_type)


def with_parameters(hypothesis, **changes):
    return replace(hypothesis, parameters={**hypothesis.parameters, **changes})


def rotated_about(direction, angle, helper=(1.0, 0.3, -0.2)):
    """`direction` tilted by `angle` radians toward an arbitrary perpendicular."""
    direction = np.asarray(direction, dtype=np.float64)
    side = np.cross(direction, helper)
    side /= np.linalg.norm(side)
    return np.cos(angle) * direction + np.sin(angle) * side


def line_distance(point, line_point, line_direction):
    offset = np.asarray(point) - line_point
    return np.linalg.norm(offset - (offset @ line_direction) * line_direction)


def assert_matches_truth(primitive_type, fitted, truth, pose, tolerance):
    """Compare parameters up to the freedom of the representation."""
    length = pose.scale * 3.0
    for key in ("normal", "axis_direction"):
        if key in truth:
            assert abs(fitted[key] @ truth[key]) == pytest.approx(1.0, abs=tolerance**2)
    for key in ("radius", "major_radius", "minor_radius"):
        if key in truth:
            assert fitted[key] == pytest.approx(truth[key], rel=tolerance)
    if "half_angle_radians" in truth:
        assert fitted["half_angle_radians"] == pytest.approx(
            truth["half_angle_radians"], abs=tolerance
        )
    if primitive_type == "cylinder":
        assert line_distance(fitted["axis_point"], truth["axis_point"], truth["axis_direction"]) < (
            tolerance * length
        )
    for key in ("center", "apex"):
        if key in truth:
            np.testing.assert_allclose(fitted[key], truth[key], rtol=0, atol=tolerance * length)


@pytest.mark.parametrize("pose_name", POSES)
@pytest.mark.parametrize("primitive_type", PRIMITIVE_TYPES)
def test_refinement_recovers_known_primitive(primitive_type, pose_name):
    pose = POSES[pose_name]
    record, truth = BUILDERS[primitive_type](pose)

    refinement = refine_hypothesis(record, seed_for(record, primitive_type))

    assert refinement.converged
    assert refinement.refined.relative_rms < 1e-9
    assert_matches_truth(primitive_type, refinement.refined.parameters, truth, pose, 1e-7)


def test_refinement_removes_the_torus_seed_bias():
    record, truth = synthetic.torus()
    seed = seed_for(record, "torus")

    refined = refine_hypothesis(record, seed).refined

    assert seed.relative_rms > 1e-6  # chord normals bias the closed-form axis
    assert refined.relative_rms < 1e-12 < seed.relative_rms
    assert refined.parameters["major_radius"] == pytest.approx(3.0, rel=1e-10)
    assert refined.parameters["minor_radius"] == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("pose_name", POSES)
@pytest.mark.parametrize("primitive_type", PRIMITIVE_TYPES)
def test_refinement_converges_from_a_perturbed_seed(primitive_type, pose_name):
    pose = POSES[pose_name]
    record, truth = BUILDERS[primitive_type](pose)
    seed = seed_for(record, primitive_type)
    p, length = seed.parameters, pose.scale * 3.0
    nudge = np.array([0.05, -0.04, 0.03]) * length

    perturbed = {
        "plane": lambda: dict(point=p["point"] + 0.1 * length * p["normal"],
                              normal=rotated_about(p["normal"], 0.1)),
        "sphere": lambda: dict(center=p["center"] + nudge, radius=p["radius"] * 1.1),
        "cylinder": lambda: dict(axis_point=p["axis_point"] + nudge,
                                 axis_direction=rotated_about(p["axis_direction"], 0.1),
                                 radius=p["radius"] * 0.9),
        "cone": lambda: dict(apex=p["apex"] + nudge,
                             axis_direction=rotated_about(p["axis_direction"], 0.08),
                             half_angle_radians=p["half_angle_radians"] * 1.15),
        "torus": lambda: dict(center=p["center"] + nudge,
                              axis_direction=rotated_about(p["axis_direction"], 0.08),
                              major_radius=p["major_radius"] * 1.05,
                              minor_radius=p["minor_radius"] * 0.95),
    }[primitive_type]()
    start = with_parameters(seed, **perturbed)

    refinement = refine_hypothesis(record, start)

    assert refinement.converged
    assert refinement.refined.relative_rms < 1e-9
    assert_matches_truth(primitive_type, refinement.refined.parameters, truth, pose, 1e-6)


def test_refined_parameters_keep_schema_conventions():
    for primitive_type in PRIMITIVE_TYPES:
        record, _ = BUILDERS[primitive_type](POSES["everything"])
        seed = seed_for(record, primitive_type)
        refined = refine_hypothesis(record, seed).refined

        assert refined.parameters.keys() == seed.parameters.keys()
        assert refined.primitive_type == primitive_type
        for key in ("normal", "axis_direction"):
            if key in refined.parameters:
                assert np.linalg.norm(refined.parameters[key]) == pytest.approx(1.0, abs=1e-14)
                assert refined.parameters[key] @ seed.parameters[key] > 0  # orientation kept
        json.dumps(refined.jsonable_parameters(), allow_nan=False)


def test_noisy_vertices_are_fitted_no_worse_than_the_seed():
    record, truth = synthetic.cylinder()
    noise = np.random.default_rng(3).normal(scale=0.01, size=record.points.shape)
    noisy = replace(record, points=record.points + noise)
    seed = seed_for(noisy, "cylinder")

    refinement = refine_hypothesis(noisy, seed, loss="linear")

    assert refinement.refined.relative_rms <= seed.relative_rms
    assert refinement.refined.parameters["radius"] == pytest.approx(1.5, abs=0.02)


def test_robust_loss_resists_outlier_vertices():
    record, truth = synthetic.cylinder()
    points = record.points.copy()
    outliers = np.arange(0, len(points), 12)
    points[outliers] += 0.6 * np.array([1.0, 0.0, 0.0]) * np.sign(np.cos(np.arange(len(outliers))))[:, None]
    dirty = replace(record, points=points)
    seed = seed_for(dirty, "cylinder")

    robust = refine_hypothesis(dirty, seed, loss="soft_l1", robust_scale=0.005).refined
    ordinary = refine_hypothesis(dirty, seed, loss="linear").refined

    error = lambda fit: abs(fit.parameters["radius"] - 1.5)  # noqa: E731
    assert error(robust) < error(ordinary)
    assert error(robust) < 0.05


def test_linear_loss_never_increases_the_rms_of_any_supplied_segment():
    manifest = json.loads((DATA_DIR.parent / "manifest.json").read_text())
    for mesh in manifest["meshes"]:
        segments = load_mesh_segments(DATA_DIR, mesh["mesh_id"]).segments
        for record in segments.values():
            for refinement in refine_segment(record, loss="linear"):
                assert refinement.refined.relative_rms <= refinement.initial.relative_rms * (
                    1 + 1e-9
                ) + 1e-15


def test_refinement_is_bitwise_repeatable():
    record, _ = synthetic.cone(POSES["everything"])

    first, second = refine_segment(record), refine_segment(record)

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.evaluations == b.evaluations
        for key in a.refined.parameters:
            assert np.array_equal(a.refined.parameters[key], b.refined.parameters[key])


def test_refinement_does_not_depend_on_pose():
    reference, _ = synthetic.torus(IDENTITY)
    posed, _ = synthetic.torus(POSES["everything"])

    a = refine_hypothesis(reference, seed_for(reference, "torus")).refined
    b = refine_hypothesis(posed, seed_for(posed, "torus")).refined

    assert b.rms_distance == pytest.approx(a.rms_distance * POSES["everything"].scale, abs=1e-12)
    assert b.parameters["major_radius"] == pytest.approx(
        a.parameters["major_radius"] * POSES["everything"].scale, rel=1e-9
    )


def test_wrong_type_stays_a_poor_fit_after_refinement():
    record, _ = synthetic.cylinder()

    by_type = {r.initial.primitive_type: r.refined for r in refine_segment(record)}

    assert by_type["cylinder"].relative_rms < 1e-9
    assert by_type["plane"].relative_rms > 0.05
    assert by_type["cylinder"].normal_rms_radians < 0.05  # chord facets vs true normals
    assert by_type["plane"].normal_rms_radians > 0.2


def test_single_triangle_plane_is_left_intact():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0.0]])
    record = build_segments("m", vertices, np.array([[0, 1, 2]]), np.zeros(1, dtype=np.int64))[0]

    (refinement,) = refine_segment(record)

    assert refinement.initial.primitive_type == "plane"
    assert refinement.refined.relative_rms == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(np.abs(refinement.refined.parameters["normal"]), [0, 0, 1], atol=1e-12)


def test_signed_distance_is_the_signed_version_of_surface_distance():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(50, 3))
    parameters = {
        "plane": {"point": np.array([0.1, 0.2, 0.3]), "normal": np.array([0.0, 0.6, 0.8])},
        "sphere": {"center": np.array([0.1, 0.0, 0.2]), "radius": 1.3},
        "cylinder": {"axis_point": np.zeros(3), "axis_direction": np.array([0.0, 0.0, 1.0]),
                     "radius": 0.9},
        "cone": {"apex": np.array([0.0, 0.0, -1.0]), "axis_direction": np.array([0.0, 0.0, 1.0]),
                 "half_angle_radians": 0.6},
        "torus": {"center": np.zeros(3), "axis_direction": np.array([0.0, 0.0, 1.0]),
                  "major_radius": 1.5, "minor_radius": 0.5},
    }
    for primitive_type, values in parameters.items():
        signed = signed_surface_distance(primitive_type, values, points)
        np.testing.assert_array_equal(np.abs(signed), surface_distance(primitive_type, values, points))
        assert (signed > 0).any() and (signed < 0).any()


# --- real data --------------------------------------------------------------

MANIFEST = json.loads((DATA_DIR.parent / "manifest.json").read_text())
ALL_SEGMENTS = [
    (m["mesh_id"], int(segment_id))
    for m in MANIFEST["meshes"]
    for segment_id in m["segment_triangle_counts"]
]


@pytest.mark.parametrize("mesh_id,segment_id", ALL_SEGMENTS)
def test_every_supplied_segment_refines_to_finite_well_formed_parameters(mesh_id, segment_id):
    record = load_mesh_segments(DATA_DIR, mesh_id).segments[segment_id]

    refinements = refine_segment(record)

    assert refinements
    for refinement in refinements:
        fit = refinement.refined
        assert np.isfinite(fit.rms_distance) and np.isfinite(fit.normal_rms_radians)
        for key, value in fit.parameters.items():
            assert np.all(np.isfinite(value))
            if key in ("normal", "axis_direction"):
                assert np.linalg.norm(value) == pytest.approx(1.0)
            if key in ("radius", "major_radius", "minor_radius"):
                assert value > 0
        if "half_angle_radians" in fit.parameters:
            assert 0 < fit.parameters["half_angle_radians"] < np.pi / 2


def refined(mesh_id, segment_id, primitive_type):
    record = load_mesh_segments(DATA_DIR, mesh_id).segments[segment_id]
    return next(
        r.refined for r in refine_segment(record) if r.initial.primitive_type == primitive_type
    )


# The supplied STL vertices are float32, so ~1e-7 relative is the noise floor.
def test_mesh_08_torus_refines_to_the_float32_noise_floor():
    fit = refined("mesh_08", 0, "torus")

    assert fit.relative_rms < 1e-7
    assert fit.parameters["major_radius"] > fit.parameters["minor_radius"] > 0


def test_mesh_02_cylinder_and_mesh_03_cone_stay_at_the_noise_floor():
    cylinder = refined("mesh_02", 0, "cylinder")
    cone = refined("mesh_03", 0, "cone")

    assert cylinder.relative_rms < 1e-7
    assert cylinder.parameters["radius"] == pytest.approx(2.0, rel=1e-7)
    assert cone.relative_rms < 1e-7
    assert cone.parameters["half_angle_radians"] == pytest.approx(np.arctan(0.25), abs=1e-7)
    np.testing.assert_allclose(cone.parameters["apex"], [0, 0, 6], atol=1e-6)


@pytest.mark.parametrize("mesh_id,segment_id", [("mesh_09", 0), ("mesh_10", 1)])
def test_freeform_segments_stay_poor_fits_after_refinement(mesh_id, segment_id):
    record = load_mesh_segments(DATA_DIR, mesh_id).segments[segment_id]

    assert min(r.refined.relative_rms for r in refine_segment(record)) > 0.05
