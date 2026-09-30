"""Closed-form initial hypotheses for each supported primitive.

Every guess is a deterministic NumPy computation (weighted PCA, linear least
squares, one SVD) on a `NormalizedSegment`; nothing iterates or draws random
numbers. The guesses are meant to seed the SciPy refinement and to give the
classifier a cheap first residual per type, not to be final fits.

* plane: PCA of the vertices.
* sphere / circles: algebraic (linear) least squares on the vertices.
* cylinder: facet normals lie on a great circle, so the axis is their weakest
  principal direction; the radius comes from a circle fit in the plane
  perpendicular to it.
* cone: facet normals lie on a small circle of the unit sphere, so the axis is
  the weakest principal direction of the mean-centred normals and the half angle
  is the arcsine of their mean axial component (used to reject cylinders and
  planes); tangent planes meet at the apex, and the vertices then give the half
  angle exactly, free of the chord bias of the facet normals.
* torus: normal lines of a surface of revolution all meet the axis, which is
  the null vector of a linear system in Plücker coordinates; a circle fit to the
  (radial, axial) profile then gives the radii and the centre.

Each guess is scored by two residuals: vertex-to-surface distance and the angle
between facet normals and the surface normal. Both are needed: two coaxial rings
of vertices (a coarse cylinder) also lie on a sphere, and only the normals tell
them apart.

A guess returns `None` when the segment is degenerate for that primitive
(too few points, normals not spanning enough directions, non-positive radius).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fitpipeline.geometry import NormalizedSegment, prepare_segment
from fitpipeline.loader import SegmentRecord
from fitpipeline.primitives import (
    PRIMITIVE_TYPES,
    Parameters,
    parameters_from_normalized,
    parameters_to_jsonable,
    surface_distance,
    surface_normal,
)

# Relative eigen/singular value below which a direction counts as absent.
_RANK_TOL = 1e-8
_TORUS_RANK_TOL = 1e-4
# lstsq cutoff (relative to the largest singular value) for point-fit designs;
# guards against coplanar/collinear samples that only look full rank through
# floating-point noise and would give an enormous radius.
_FIT_RCOND = 1e-6
# Half angles this close to 0 (cylinder) or pi/2 (plane) are not cones.
_MIN_HALF_ANGLE = 1e-3


@dataclass(frozen=True)
class Hypothesis:
    """An initial guess for one primitive type, in the input frame and units."""

    primitive_type: str
    parameters: Parameters
    rms_distance: float  # area-weighted RMS vertex-to-surface distance, input units
    relative_rms: float  # rms_distance divided by the segment's normalization scale
    normal_rms_radians: float  # area-weighted RMS angle between facet and surface normals

    def jsonable_parameters(self) -> dict[str, list[float] | float]:
        return parameters_to_jsonable(self.parameters)


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    """Flip so the largest-magnitude component is positive (SVD/eigh signs are arbitrary)."""
    pivot = np.argmax(np.round(np.abs(vector), 9))
    return vector if vector[pivot] >= 0 else -vector


def _orient(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    alignment = vector @ reference
    if alignment > 1e-9:
        return vector
    if alignment < -1e-9:
        return -vector
    return _canonical_sign(vector)


def _normal_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in [0, pi/2] between row-wise unit vectors, ignoring sign."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.abs((a * b).sum(axis=1)))


def _second_moment(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (vectors * weights[:, None]).T @ vectors


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.zeros(3)
    helper[np.argmin(np.abs(axis))] = 1.0
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _fit_circle(
    u: np.ndarray, v: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float] | None:
    """Weighted algebraic circle fit; returns (centre_u, centre_v, radius)."""
    design = np.column_stack([2 * u, 2 * v, np.ones_like(u)])
    target = u * u + v * v
    root_w = np.sqrt(weights)
    solution, _, rank, _ = np.linalg.lstsq(
        design * root_w[:, None], target * root_w, rcond=_FIT_RCOND
    )
    if rank < 3:
        return None
    cu, cv, offset = solution
    radius_squared = offset + cu * cu + cv * cv
    if not radius_squared > 0:
        return None
    return float(cu), float(cv), float(np.sqrt(radius_squared))


def _guess_plane(segment: NormalizedSegment) -> Parameters | None:
    x, w = segment.vertices, segment.vertex_weights
    centroid = w @ x
    eigenvalues, eigenvectors = np.linalg.eigh(_second_moment(x - centroid, w))
    if eigenvalues[1] <= _RANK_TOL * eigenvalues[2]:  # collinear or a single point
        return None
    mean_normal = segment.facet_weights @ segment.facet_normals
    return {"point": centroid, "normal": _orient(eigenvectors[:, 0], mean_normal)}


def _guess_sphere(segment: NormalizedSegment) -> Parameters | None:
    x, w = segment.vertices, segment.vertex_weights
    # |x|^2 = 2 c.x + (r^2 - |c|^2) is linear in (c, r^2 - |c|^2).
    design = np.column_stack([2 * x, np.ones(len(x))])
    root_w = np.sqrt(w)
    solution, _, rank, _ = np.linalg.lstsq(
        design * root_w[:, None], (x * x).sum(axis=1) * root_w, rcond=_FIT_RCOND
    )
    if rank < 4:  # coplanar samples cannot fix a centre
        return None
    center = solution[:3]
    radius_squared = solution[3] + center @ center
    if not radius_squared > 0:
        return None
    return {"center": center, "radius": float(np.sqrt(radius_squared))}


def _guess_cylinder(segment: NormalizedSegment) -> Parameters | None:
    x, w = segment.vertices, segment.vertex_weights
    eigenvalues, eigenvectors = np.linalg.eigh(
        _second_moment(segment.facet_normals, segment.facet_weights)
    )
    if eigenvalues[1] <= _RANK_TOL:  # normals all (nearly) parallel: a plane
        return None
    axis = _canonical_sign(eigenvectors[:, 0])
    first, second = perpendicular_basis(axis)
    circle = _fit_circle(x @ first, x @ second, w)
    if circle is None:
        return None
    cu, cv, radius = circle
    axis_point = cu * first + cv * second + (w @ (x @ axis)) * axis
    return {"axis_point": axis_point, "axis_direction": axis, "radius": radius}


def _guess_cone(segment: NormalizedSegment) -> Parameters | None:
    x, w = segment.vertices, segment.vertex_weights
    normals, weights = segment.facet_normals, segment.facet_weights
    mean_normal = weights @ normals
    eigenvalues, eigenvectors = np.linalg.eigh(_second_moment(normals - mean_normal, weights))
    if eigenvalues[1] <= _RANK_TOL:
        return None
    axis = eigenvectors[:, 0]
    normal_half_angle = float(np.arcsin(min(abs(weights @ (normals @ axis)), 1.0)))
    if not _MIN_HALF_ANGLE < normal_half_angle < np.pi / 2 - _MIN_HALF_ANGLE:
        return None

    # Every tangent plane passes through the apex: n_i . apex = n_i . c_i.
    root_w = np.sqrt(weights)
    apex, _, rank, _ = np.linalg.lstsq(
        normals * root_w[:, None],
        (normals * segment.facet_centroids).sum(axis=1) * root_w,
        rcond=None,
    )
    if rank < 3:
        return None
    axis = _orient(axis, w @ x - apex)  # point the axis from the apex toward the segment

    # Facet planes are chords, so their normals slightly understate the half
    # angle of a tessellated cone; the vertices lie on the true cone. With the
    # apex and axis fixed, r = h tan(a) gives the angle in closed form.
    offsets = x - apex
    axial = offsets @ axis
    radial = np.linalg.norm(offsets - axial[:, None] * axis, axis=1)
    if not w @ axial**2 > 0:
        return None
    half_angle = float(np.arctan((w @ (axial * radial)) / (w @ axial**2)))
    if not _MIN_HALF_ANGLE < half_angle < np.pi / 2 - _MIN_HALF_ANGLE:
        return None
    return {"apex": apex, "axis_direction": axis, "half_angle_radians": half_angle}


def _guess_torus(segment: NormalizedSegment) -> Parameters | None:
    x, w = segment.vertices, segment.vertex_weights
    centroids, normals = segment.facet_centroids, segment.facet_normals
    if len(normals) < 5:
        return None

    # The normal line (direction n, moment c x n) meets the axis (direction a,
    # moment b) iff a.(c x n) + b.n = 0, linear in the 6-vector (a, b).
    system = np.hstack([np.cross(centroids, normals), normals]) * np.sqrt(
        segment.facet_weights
    )[:, None]
    _, singular_values, right_vectors = np.linalg.svd(system, full_matrices=True)
    singular_values = np.pad(singular_values, (0, 6 - len(singular_values)))
    if singular_values[4] <= _TORUS_RANK_TOL * singular_values[0]:  # axis not unique
        return None
    direction, moment = right_vectors[-1, :3], right_vectors[-1, 3:]
    length = np.linalg.norm(direction)
    if not length > 0:
        return None
    axis = direction / length
    moment = (moment - (moment @ axis) * axis) / length  # enforce a.b = 0
    axis_point = np.cross(axis, moment)  # closest axis point to the segment centroid
    axis = _canonical_sign(axis)

    offsets = x - axis_point
    axial = offsets @ axis
    radial = np.linalg.norm(offsets - axial[:, None] * axis, axis=1)
    circle = _fit_circle(radial, axial, w)
    if circle is None:
        return None
    major_radius, axial_center, minor_radius = circle
    if not major_radius > 0:
        return None
    return {
        "center": axis_point + axial_center * axis,
        "axis_direction": axis,
        "major_radius": major_radius,
        "minor_radius": minor_radius,
    }


_GUESSERS = {
    "plane": _guess_plane,
    "cylinder": _guess_cylinder,
    "cone": _guess_cone,
    "sphere": _guess_sphere,
    "torus": _guess_torus,
}


def _is_finite(parameters: Parameters) -> bool:
    return all(np.isfinite(value).all() for value in parameters.values())


def score_parameters(
    primitive_type: str, normalized_parameters: Parameters, segment: NormalizedSegment
) -> tuple[float, float]:
    """(relative RMS vertex distance, RMS facet-normal angle in radians) of a fit.

    Parameters are in the segment's normalized frame; both averages are weighted by area.
    """
    distances = surface_distance(primitive_type, normalized_parameters, segment.vertices)
    relative_rms = float(np.sqrt(segment.vertex_weights @ distances**2))
    surface_normals = surface_normal(
        primitive_type, normalized_parameters, segment.facet_centroids
    )
    angles = _normal_angle(segment.facet_normals, surface_normals)
    return relative_rms, float(np.sqrt(segment.facet_weights @ angles**2))


def propose_hypotheses(record: SegmentRecord) -> list[Hypothesis]:
    """Initial guesses for every primitive type that is computable for the segment.

    Ordered like `PRIMITIVE_TYPES`; ranking them is left to the classifier.
    Raises `InsufficientGeometryError` if the segment has no usable area.
    """
    segment = prepare_segment(record)
    scale = segment.normalization.scale
    hypotheses = []
    for primitive_type in PRIMITIVE_TYPES:
        normalized = _GUESSERS[primitive_type](segment)
        if normalized is None or not _is_finite(normalized):
            continue
        relative_rms, normal_rms = score_parameters(primitive_type, normalized, segment)
        hypotheses.append(
            Hypothesis(
                primitive_type=primitive_type,
                parameters=parameters_from_normalized(
                    primitive_type, normalized, segment.normalization
                ),
                rms_distance=relative_rms * scale,
                relative_rms=relative_rms,
                normal_rms_radians=normal_rms,
            )
        )
    return hypotheses
