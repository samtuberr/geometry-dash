"""Final classification of a segment: a supported primitive, or `unresolved`.

Every computable primitive type is seeded (`hypotheses.py`) and refined
(`refinement.py`); this module then judges the refined candidates by their
residuals and picks at most one.

A candidate is *accepted* only if all of these hold:

* the area-weighted RMS vertex-to-surface distance is below `position_tolerance`
  and the largest one below `max_position_tolerance`, both relative to the
  segment's RMS extent (the natural scale of the normalized frame);
* the RMS angle between facet normals and the surface normal at the facet
  centroid is below a tessellation-aware bound: a flat facet inscribed in a
  curved surface is tilted by up to half the angle between neighbouring facets,
  so the bound is half of the largest dihedral angle inside the segment (with a
  small floor for flat segments). Vertices alone cannot be trusted here, because
  two coaxial rings of a coarse cylinder also lie exactly on a sphere;
* the segment has enough vertices to over-determine the primitive, and its
  radii are resolved (a "sphere" a thousand times wider than the segment is a
  plane).

Several accepted candidates are common (a plane is a huge sphere, a cylinder is
a narrow cone, ...). Candidates are walked from the fewest parameters to the most
and a more general one replaces the current choice only if it lowers the normal
error clearly (`SIMPLICITY_MARGIN`), so the simplest adequate primitive wins.
If no candidate is accepted the segment is `unresolved`, with the closest
candidate kept for the fit report.

`confidence` is a heuristic score in [0, 1], not a calibrated probability: for a
primitive it grows with the position-residual margin and the evidence
(vertex count) and shrinks when a non-nested rival type is also accepted; for
`unresolved` it grows with how far the best candidate misses its tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fitpipeline.geometry import InsufficientGeometryError, prepare_segment
from fitpipeline.hypotheses import propose_hypotheses
from fitpipeline.loader import SegmentRecord
from fitpipeline.primitives import surface_distance
from fitpipeline.refinement import Refinement, refine_hypothesis

# Free parameters of each primitive's refinement chart; also its complexity rank.
DEGREES_OF_FREEDOM = {"plane": 3, "sphere": 4, "cylinder": 5, "cone": 6, "torus": 7}
# The more general types that contain each type as a limiting case; they fit
# whenever it does, so they are not counted as rivals of it.
_LIMIT_OF = {
    "plane": {"sphere", "cylinder", "cone", "torus"},
    "sphere": {"torus"},
    "cylinder": {"cone", "torus"},
    "cone": set(),
    "torus": set(),
}

POSITION_TOLERANCE = 1e-3  # relative RMS vertex distance
MAX_POSITION_TOLERANCE = 1e-2  # relative largest vertex distance
MIN_NORMAL_TOLERANCE = np.radians(0.5)  # floor of the tessellation-aware bound
NORMAL_TOLERANCE_FRACTION = 0.5  # of the largest dihedral angle in the segment
SIMPLICITY_MARGIN = 0.8  # a more general type must reduce the normal error to this fraction
MIN_NORMAL_IMPROVEMENT = np.radians(0.005)  # ... and by at least this much in absolute terms
EXTRA_VERTICES = 3  # a curved type needs at least dof + this many vertices
MAX_RELATIVE_RADIUS = 1e3  # radii above this multiple of the segment extent are unresolved


@dataclass(frozen=True)
class Candidate:
    """One refined primitive hypothesis with its verdict."""

    refinement: Refinement
    relative_rms: float
    relative_max: float
    normal_rms_radians: float
    position_ratio: float  # worst position residual over its tolerance; <= 1 passes
    normal_ratio: float  # normal residual over its tolerance; <= 1 passes
    problems: tuple[str, ...]  # non-residual reasons for rejection

    @property
    def primitive_type(self) -> str:
        return self.refinement.refined.primitive_type

    @property
    def accepted(self) -> bool:
        return self.position_ratio <= 1 and self.normal_ratio <= 1 and not self.problems

    @property
    def violation(self) -> float:
        return max(self.position_ratio, self.normal_ratio)


@dataclass(frozen=True)
class Classification:
    kind: str  # "primitive" or "unresolved"
    primitive_type: str | None
    confidence: float
    reason: str
    best: Candidate | None  # chosen candidate, or the closest one when unresolved
    candidates: tuple[Candidate, ...] = field(default=())
    scale: float = 0.0  # RMS extent of the segment, input units
    normal_tolerance_radians: float = 0.0


def max_dihedral_angle(record: SegmentRecord) -> float:
    """Largest angle (radians) between the normals of triangles sharing an edge in the segment."""
    triangles = record.triangles
    edges = np.sort(
        np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
    )
    owners = np.tile(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    shared = np.flatnonzero((edges[1:] == edges[:-1]).all(axis=1))
    if len(shared) == 0:
        return 0.0
    first, second = record.triangle_normals[owners[shared]], record.triangle_normals[owners[shared + 1]]
    valid = (np.linalg.norm(first, axis=1) > 0) & (np.linalg.norm(second, axis=1) > 0)
    if not valid.any():
        return 0.0
    cosines = np.clip((first[valid] * second[valid]).sum(axis=1), -1.0, 1.0)
    return float(np.arccos(cosines).max())


def _radii(parameters) -> list[float]:
    return [float(parameters[key]) for key in ("radius", "major_radius", "minor_radius") if key in parameters]


def _evaluate(
    refinement: Refinement,
    record: SegmentRecord,
    scale: float,
    normal_tolerance: float,
    position_tolerance: float,
    max_position_tolerance: float,
) -> Candidate:
    hypothesis = refinement.refined
    primitive_type = hypothesis.primitive_type
    distances = surface_distance(primitive_type, hypothesis.parameters, record.points)
    relative_max = float(distances.max() / scale)

    problems = []
    dof = DEGREES_OF_FREEDOM[primitive_type]
    if primitive_type != "plane" and record.point_count < dof + EXTRA_VERTICES:
        problems.append(
            f"only {record.point_count} vertices for {dof} free parameters"
        )
    if any(radius > MAX_RELATIVE_RADIUS * scale for radius in _radii(hypothesis.parameters)):
        problems.append("radius is not resolved (surface indistinguishable from a plane)")

    return Candidate(
        refinement=refinement,
        relative_rms=hypothesis.relative_rms,
        relative_max=relative_max,
        normal_rms_radians=hypothesis.normal_rms_radians,
        position_ratio=max(
            hypothesis.relative_rms / position_tolerance, relative_max / max_position_tolerance
        ),
        normal_ratio=hypothesis.normal_rms_radians / normal_tolerance,
        problems=tuple(problems),
    )


def _choose(accepted: list[Candidate]) -> Candidate:
    """Simplest accepted candidate, replaced only by a clearly better-fitting general one."""
    ordered = sorted(accepted, key=lambda c: DEGREES_OF_FREEDOM[c.primitive_type])
    chosen = ordered[0]
    for challenger in ordered[1:]:
        gain = chosen.normal_rms_radians - challenger.normal_rms_radians
        if (
            challenger.normal_rms_radians < SIMPLICITY_MARGIN * chosen.normal_rms_radians
            and gain > MIN_NORMAL_IMPROVEMENT
        ):
            chosen = challenger
    return chosen


def _describe(candidate: Candidate) -> str:
    return (
        f"{candidate.primitive_type}: vertex RMS {candidate.relative_rms:.1e} of extent, "
        f"normal RMS {np.degrees(candidate.normal_rms_radians):.2f} deg"
    )


def _primitive_confidence(chosen: Candidate, rivals: list[Candidate], vertex_count: int) -> float:
    position_margin = float(np.clip(-np.log10(max(chosen.position_ratio, 1e-12)) / 3.0, 0.0, 1.0))
    dof = DEGREES_OF_FREEDOM[chosen.primitive_type]
    if chosen.primitive_type == "plane":
        evidence = 1.0 if vertex_count >= 4 else 0.5
    else:
        evidence = min(1.0, (vertex_count - dof) / (2.0 * dof))
    separation = 1.0
    for rival in rivals:
        # 0 when the rival explains the normals as well as the winner, 1 when it is far worse.
        floor = max(rival.normal_rms_radians, 1e-12)
        separation = min(separation, float(np.clip(1.0 - chosen.normal_rms_radians / floor, 0.0, 1.0)))
    confidence = (0.6 + 0.39 * position_margin) * (0.5 + 0.5 * evidence) * (0.6 + 0.4 * separation)
    return round(float(np.clip(confidence, 0.0, 1.0)), 3)


def classify_segment(
    record: SegmentRecord,
    *,
    position_tolerance: float = POSITION_TOLERANCE,
    max_position_tolerance: float = MAX_POSITION_TOLERANCE,
) -> Classification:
    """Classify one segment; never raises for degenerate geometry (returns `unresolved`)."""
    try:
        scale = prepare_segment(record).normalization.scale
        hypotheses = propose_hypotheses(record)
    except InsufficientGeometryError as error:
        return Classification("unresolved", None, 0.5, f"insufficient evidence: {error}", None)

    normal_tolerance = max(
        MIN_NORMAL_TOLERANCE, NORMAL_TOLERANCE_FRACTION * max_dihedral_angle(record)
    )
    candidates = []
    for hypothesis in hypotheses:
        try:
            refinement = refine_hypothesis(record, hypothesis)
        except (ValueError, np.linalg.LinAlgError):
            continue  # numerical failure: this type simply gets no candidate
        candidates.append(
            _evaluate(
                refinement, record, scale, normal_tolerance,
                position_tolerance, max_position_tolerance,
            )
        )
    if not candidates:
        return Classification(
            "unresolved", None, 0.5,
            "insufficient evidence: no primitive type could be fitted numerically",
            None, scale=scale, normal_tolerance_radians=normal_tolerance,
        )

    accepted = [candidate for candidate in candidates if candidate.accepted]
    if not accepted:
        closest = min(candidates, key=lambda c: c.violation)
        if closest.problems:
            reason = f"closest candidate {closest.primitive_type} rejected: {'; '.join(closest.problems)}"
            confidence = 0.5
        else:
            reason = (
                f"no supported primitive explains the segment; closest is {_describe(closest)} "
                f"(tolerance: vertex RMS {position_tolerance:.0e} of extent, normal RMS "
                f"{np.degrees(normal_tolerance):.2f} deg)"
            )
            confidence = 0.5 + 0.45 * float(np.clip(np.log10(closest.violation) / 2.0, 0.0, 1.0))
        return Classification(
            "unresolved", None, round(confidence, 3), reason, closest, tuple(candidates),
            scale, normal_tolerance,
        )

    chosen = _choose(accepted)
    rivals = [
        candidate for candidate in accepted
        if candidate is not chosen and candidate.primitive_type not in _LIMIT_OF[chosen.primitive_type]
    ]
    reason = f"{_describe(chosen)}, within tolerance"
    if rivals:
        reason += "; rival types also fit the vertices but explain the facet normals worse: " + ", ".join(
            _describe(rival) for rival in rivals
        )
    return Classification(
        "primitive", chosen.primitive_type,
        _primitive_confidence(chosen, rivals, record.point_count),
        reason, chosen, tuple(candidates), scale, normal_tolerance,
    )
