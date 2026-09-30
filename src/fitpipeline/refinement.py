"""Nonlinear refinement of a primitive hypothesis with `scipy.optimize.least_squares`.

The closed-form guesses of `hypotheses.py` are algebraic or normal-based, so they
carry small biases (chord normals of a tessellated torus, an algebraic rather
than geometric circle fit). Here the objective is the geometric one: the
area-weighted signed vertex-to-surface distances from `primitives.py`, minimised
over the parameters of the guessed primitive. They are signed because the absolute
value has a kink at the surface that spoils the finite-difference Jacobian once
residuals get small.

Everything runs in the segment's normalized frame and the result is mapped back
to the input frame. Parameters are optimised in a minimal, unconstrained-looking
chart around the seed, so no gauge freedom is left for the optimiser to wander in:

* directions are `normalize(seed + t1 * e1 + t2 * e2)` with `e1, e2` spanning the
  plane perpendicular to the seed direction (2 parameters, always unit length);
* a cylinder axis point may only slide across the axis, a plane point only along
  the normal, since the other components do not change the surface;
* radii and the cone half angle are bounded away from 0 (and pi/2).

The optimiser only accepts steps that lower its (robust) cost, so a refinement
never ends up worse than its seed under that cost. With the default robust loss
the *plain* RMS can still rise slightly when a few vertices are far off, which
is the point of using it; pass `loss="linear"` for ordinary least squares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from fitpipeline.geometry import NormalizedSegment, prepare_segment
from fitpipeline.hypotheses import (
    Hypothesis,
    perpendicular_basis,
    propose_hypotheses,
    score_parameters,
)
from fitpipeline.loader import SegmentRecord
from fitpipeline.primitives import (
    Parameters,
    parameters_from_normalized,
    parameters_to_normalized,
    signed_surface_distance,
)

# Lengths are in units of the segment's RMS extent, so these are relative bounds.
_MIN_LENGTH = 1e-6
_MIN_HALF_ANGLE = 1e-3
# Residuals (normalized distances) below this count as inliers of the robust loss.
DEFAULT_ROBUST_SCALE = 1e-2
_TOLERANCE = 1e-12
_MAX_EVALUATIONS = 200


@dataclass(frozen=True)
class Refinement:
    """A seed hypothesis and its refined counterpart, both in the input frame."""

    initial: Hypothesis
    refined: Hypothesis
    converged: bool  # least_squares stopped on a tolerance rather than on max_nfev
    evaluations: int  # residual function evaluations
    message: str


@dataclass(frozen=True)
class _Chart:
    """Local parametrization of a primitive: theta -> parameters."""

    theta0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    build: Callable[[np.ndarray], Parameters]


def _tilted(axis: np.ndarray, basis: tuple[np.ndarray, np.ndarray], t1: float, t2: float):
    direction = axis + t1 * basis[0] + t2 * basis[1]
    return direction / np.linalg.norm(direction)


def _bounds(theta0: list[float], lower: dict[int, float], upper: dict[int, float]):
    low = np.full(len(theta0), -np.inf)
    high = np.full(len(theta0), np.inf)
    for index, bound in lower.items():
        low[index] = bound
    for index, bound in upper.items():
        high[index] = bound
    # A seed exactly on (or, from a caller's hand-made guess, beyond) a bound must
    # start strictly inside the box for the trust-region solver.
    x0 = np.array(theta0, dtype=np.float64)
    inside = np.isfinite(low) | np.isfinite(high)
    margin = 1e-9 * np.maximum(1.0, np.abs(x0))
    x0[inside] = np.clip(x0[inside], low[inside] + margin[inside], high[inside] - margin[inside])
    return x0, low, high


def _plane_chart(p: Parameters) -> _Chart:
    point, normal = p["point"], p["normal"]
    basis = perpendicular_basis(normal)
    x0, low, high = _bounds([0.0, 0.0, 0.0], {}, {})
    return _Chart(
        x0, low, high,
        lambda t: {"point": point + t[0] * normal, "normal": _tilted(normal, basis, t[1], t[2])},
    )


def _sphere_chart(p: Parameters) -> _Chart:
    center = p["center"]
    x0, low, high = _bounds([*center, p["radius"]], {3: _MIN_LENGTH}, {})
    return _Chart(x0, low, high, lambda t: {"center": t[:3], "radius": t[3]})


def _cylinder_chart(p: Parameters) -> _Chart:
    origin, axis = p["axis_point"], p["axis_direction"]
    basis = perpendicular_basis(axis)
    x0, low, high = _bounds([0.0, 0.0, 0.0, 0.0, p["radius"]], {4: _MIN_LENGTH}, {})
    return _Chart(
        x0, low, high,
        lambda t: {
            "axis_point": origin + t[0] * basis[0] + t[1] * basis[1],
            "axis_direction": _tilted(axis, basis, t[2], t[3]),
            "radius": t[4],
        },
    )


def _cone_chart(p: Parameters) -> _Chart:
    axis = p["axis_direction"]
    basis = perpendicular_basis(axis)
    x0, low, high = _bounds(
        [*p["apex"], 0.0, 0.0, p["half_angle_radians"]],
        {5: _MIN_HALF_ANGLE},
        {5: np.pi / 2 - _MIN_HALF_ANGLE},
    )
    return _Chart(
        x0, low, high,
        lambda t: {
            "apex": t[:3],
            "axis_direction": _tilted(axis, basis, t[3], t[4]),
            "half_angle_radians": t[5],
        },
    )


def _torus_chart(p: Parameters) -> _Chart:
    axis = p["axis_direction"]
    basis = perpendicular_basis(axis)
    x0, low, high = _bounds(
        [*p["center"], 0.0, 0.0, p["major_radius"], p["minor_radius"]],
        {5: _MIN_LENGTH, 6: _MIN_LENGTH},
        {},
    )
    return _Chart(
        x0, low, high,
        lambda t: {
            "center": t[:3],
            "axis_direction": _tilted(axis, basis, t[3], t[4]),
            "major_radius": t[5],
            "minor_radius": t[6],
        },
    )


_CHARTS: dict[str, Callable[[Parameters], _Chart]] = {
    "plane": _plane_chart,
    "cylinder": _cylinder_chart,
    "cone": _cone_chart,
    "sphere": _sphere_chart,
    "torus": _torus_chart,
}


def _refine(
    segment: NormalizedSegment, hypothesis: Hypothesis, loss: str, robust_scale: float
) -> Refinement:
    primitive_type = hypothesis.primitive_type
    normalization = segment.normalization
    seed = parameters_to_normalized(primitive_type, hypothesis.parameters, normalization)
    chart = _CHARTS[primitive_type](seed)
    root_weights = np.sqrt(segment.vertex_weights)

    def residuals(theta: np.ndarray) -> np.ndarray:
        parameters = chart.build(theta)
        return root_weights * signed_surface_distance(primitive_type, parameters, segment.vertices)

    result = least_squares(
        residuals,
        chart.theta0,
        bounds=(chart.lower, chart.upper),
        method="trf",
        loss=loss,
        f_scale=robust_scale,
        x_scale="jac",
        ftol=_TOLERANCE,
        xtol=_TOLERANCE,
        gtol=_TOLERANCE,
        max_nfev=_MAX_EVALUATIONS,
    )

    fitted = chart.build(result.x)
    finite = all(np.isfinite(value).all() for value in fitted.values())
    if not finite:
        return Refinement(hypothesis, hypothesis, False, result.nfev, "non-finite result; kept seed")

    relative_rms, normal_rms = score_parameters(primitive_type, fitted, segment)
    refined = Hypothesis(
        primitive_type=primitive_type,
        parameters=parameters_from_normalized(primitive_type, fitted, normalization),
        rms_distance=relative_rms * normalization.scale,
        relative_rms=relative_rms,
        normal_rms_radians=normal_rms,
    )
    return Refinement(
        initial=hypothesis,
        refined=refined,
        converged=result.status > 0,
        evaluations=int(result.nfev),
        message=result.message,
    )


def refine_hypothesis(
    record: SegmentRecord,
    hypothesis: Hypothesis,
    *,
    loss: str = "soft_l1",
    robust_scale: float = DEFAULT_ROBUST_SCALE,
) -> Refinement:
    """Refine one hypothesis (input frame) against the vertices of `record`.

    `robust_scale` is in units of the segment's RMS extent and only matters for
    non-linear losses. Raises `InsufficientGeometryError` for a degenerate segment.
    """
    return _refine(prepare_segment(record), hypothesis, loss, robust_scale)


def refine_segment(
    record: SegmentRecord,
    *,
    loss: str = "soft_l1",
    robust_scale: float = DEFAULT_ROBUST_SCALE,
) -> list[Refinement]:
    """Propose and refine every computable primitive type; ordered like `PRIMITIVE_TYPES`."""
    segment = prepare_segment(record)
    return [
        _refine(segment, hypothesis, loss, robust_scale)
        for hypothesis in propose_hypotheses(record)
    ]
