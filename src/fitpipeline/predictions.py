"""Assemble and write `predictions.json`: one entry per (mesh_id, segment_id).

Each entry follows `assignment/README.md`: `kind`, `primitive_type`,
`confidence`, `reason` and a `fit` object with `status`, `parameters`, `notes`
plus a `residual` summary. Fit status:

* `fitted`: accepted primitive whose refinement converged;
* `partial`: accepted primitive, but the optimiser stopped on its evaluation
  budget instead of a tolerance, so the estimate is uncertain;
* `failed`: attempted but no usable estimate, including every `unresolved`
  segment (the closest attempted type is named in `notes`; its poor parameters
  are withheld rather than reported as if they were a fit).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from fitpipeline.classifier import Candidate, Classification, classify_segment
from fitpipeline.loader import MeshData, SegmentRecord, load_all_meshes
from fitpipeline.primitives import PRIMITIVE_TYPES, parameters_to_jsonable

_OBJECTIVE = (
    "scipy least_squares (trf, soft_l1 loss) on signed vertex-to-surface distances, "
    "weighted by one third of the incident triangle area, in a frame normalized by the "
    "segment's RMS extent and mapped back to the input frame; seeded by closed-form "
    "PCA/linear-least-squares guesses"
)


def _residual(candidate: Candidate, scale: float) -> dict:
    hypothesis = candidate.refinement.refined
    return {
        "units": "input mesh length units; normal angle in degrees",
        "sampling": "mesh vertices (RMS weighted by one third of incident triangle area) "
        "and facet centroids (normals, area weighted)",
        "vertex_rms_distance": float(hypothesis.rms_distance),
        "vertex_max_distance": float(candidate.relative_max * scale),
        "relative_vertex_rms": float(candidate.relative_rms),
        "facet_normal_rms_degrees": float(np.degrees(candidate.normal_rms_radians)),
    }


def _fit_for(classification: Classification) -> dict:
    best = classification.best
    if best is None:
        return {
            "status": "failed",
            "parameters": None,
            "notes": "No primitive could be fitted: " + classification.reason,
        }
    residual = _residual(best, classification.scale)
    refinement = best.refinement
    if classification.kind == "unresolved":
        return {
            "status": "failed",
            "parameters": None,
            "notes": (
                f"Attempted {best.primitive_type}, revised to unresolved: {classification.reason}. "
                "Parameters withheld because the fit does not follow the geometry. " + _OBJECTIVE + "."
            ),
            "residual": residual,
        }
    converged = refinement.converged
    return {
        "status": "fitted" if converged else "partial",
        "parameters": parameters_to_jsonable(refinement.refined.parameters),
        "notes": (
            f"{_OBJECTIVE}. "
            + (
                f"Converged after {refinement.evaluations} residual evaluations."
                if converged
                else f"Stopped on the {refinement.evaluations}-evaluation budget without converging; "
                "treat as approximate."
            )
            + " Plane normals and cylinder/torus axes are unit vectors with arbitrary sign; "
            "the cone axis points from the apex toward the segment."
        ),
        "residual": residual,
    }


def build_prediction(record: SegmentRecord) -> dict:
    classification = classify_segment(record)
    return {
        "mesh_id": record.mesh_id,
        "segment_id": record.segment_id,
        "kind": classification.kind,
        "primitive_type": classification.primitive_type,
        "confidence": classification.confidence,
        "reason": classification.reason,
        "fit": _fit_for(classification),
    }


def build_predictions(meshes: dict[str, MeshData]) -> list[dict]:
    """Entries ordered by mesh_id then segment_id."""
    return [
        build_prediction(mesh.segments[segment_id])
        for mesh_id, mesh in sorted(meshes.items())
        for segment_id in sorted(mesh.segments)
    ]


def _check_finite(value, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{where}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: non-finite number")


def validate_predictions(predictions: list[dict], meshes: dict[str, MeshData]) -> None:
    """Raise `ValueError` unless the entries match the required schema exactly once per segment."""
    expected = {(m, s) for m, mesh in meshes.items() for s in mesh.segments}
    seen = [(p["mesh_id"], p["segment_id"]) for p in predictions]
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError("predictions must contain exactly one entry per (mesh_id, segment_id)")
    for p in predictions:
        where = f"{p['mesh_id']}/segment {p['segment_id']}"
        if p["kind"] not in ("primitive", "unresolved"):
            raise ValueError(f"{where}: bad kind {p['kind']!r}")
        if (p["kind"] == "primitive") != (p["primitive_type"] in PRIMITIVE_TYPES):
            raise ValueError(f"{where}: primitive_type {p['primitive_type']!r} inconsistent with kind")
        if not (isinstance(p["confidence"], float) and 0.0 <= p["confidence"] <= 1.0):
            raise ValueError(f"{where}: confidence must be a float in [0, 1]")
        if not p["reason"]:
            raise ValueError(f"{where}: missing reason")
        fit = p["fit"]
        if fit["status"] not in ("fitted", "partial", "failed", "not_applicable"):
            raise ValueError(f"{where}: bad fit status {fit['status']!r}")
        if p["kind"] == "primitive" and fit["status"] == "not_applicable":
            raise ValueError(f"{where}: a primitive needs a fitting attempt")
        _check_finite(fit, f"{where}.fit")


def predict_dataset(data_dir: Path | str) -> list[dict]:
    """Classify and fit every segment under `data_dir` and validate the entries."""
    meshes = load_all_meshes(data_dir)
    predictions = build_predictions(meshes)
    validate_predictions(predictions, meshes)
    return predictions


def save_predictions(predictions: list[dict], output_path: Path | str) -> None:
    Path(output_path).write_text(
        json.dumps({"predictions": predictions}, indent=2, allow_nan=False) + "\n"
    )


def write_predictions(data_dir: Path | str, output_path: Path | str) -> list[dict]:
    """Classify and fit every segment under `data_dir`, validate, and write `output_path`."""
    predictions = predict_dataset(data_dir)
    save_predictions(predictions, output_path)
    return predictions
