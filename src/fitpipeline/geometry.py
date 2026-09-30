"""Basic per-segment calculations: area weights, sample points and scaling.

Turns a `SegmentRecord` into a `NormalizedSegment`: everything is expressed in
a frame centred on the segment's area centroid and scaled to unit RMS extent,
so downstream numerics (SVD, least squares, tolerances) do not depend on where
the mesh sits or what units it uses. Results are mapped back to the input frame
with the `Normalization` that produced them.

Two sample sets are exposed, both weighted by triangle area so that uneven
tessellation does not bias the fit:

* vertices, weighted by one third of the area of each incident triangle. They
  lie on the underlying surface, so they are the positional samples;
* facet centroids with facet normals, used for orientation (Gauss-map) analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fitpipeline.loader import SegmentRecord


class InsufficientGeometryError(ValueError):
    """The segment has no usable area or extent to compute anything from."""


@dataclass(frozen=True)
class Normalization:
    """Similarity transform: normalized = (original - center) / scale."""

    center: np.ndarray  # (3,) float64, in the input frame
    scale: float  # input-frame length that maps to 1.0

    def points_to_normalized(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - self.center) / self.scale

    def points_from_normalized(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) * self.scale + self.center

    def lengths_to_normalized(self, lengths: np.ndarray | float) -> np.ndarray | float:
        return lengths / self.scale

    def lengths_from_normalized(self, lengths: np.ndarray | float) -> np.ndarray | float:
        return lengths * self.scale


@dataclass(frozen=True)
class NormalizedSegment:
    """Area-weighted samples of one segment in its normalized frame."""

    mesh_id: str
    segment_id: int
    normalization: Normalization
    vertices: np.ndarray  # (V, 3) float64
    vertex_weights: np.ndarray  # (V,) float64, positive, sum to 1
    facet_centroids: np.ndarray  # (F, 3) float64
    facet_normals: np.ndarray  # (F, 3) float64, unit vectors
    facet_weights: np.ndarray  # (F,) float64, positive, sum to 1


def prepare_segment(record: SegmentRecord) -> NormalizedSegment:
    """Normalize a segment and build its weighted vertex and facet samples."""
    if not np.isfinite(record.points).all():
        raise InsufficientGeometryError(
            f"{record.mesh_id}/segment {record.segment_id}: non-finite vertex coordinates"
        )
    areas = record.triangle_areas
    total_area = float(areas.sum())
    if not total_area > 0:
        raise InsufficientGeometryError(
            f"{record.mesh_id}/segment {record.segment_id}: zero total area"
        )

    corners = record.points[record.triangles]  # (T, 3, 3)
    centroids = corners.mean(axis=1)
    area_weights = areas / total_area

    # Offset by a vertex before summing so far-from-origin meshes keep precision.
    reference = record.points[0]
    center = reference + area_weights @ (centroids - reference)

    mean_square_extent = area_weights @ ((corners - center) ** 2).sum(axis=2).mean(axis=1)
    scale = float(np.sqrt(mean_square_extent))
    if not scale > 0:
        raise InsufficientGeometryError(
            f"{record.mesh_id}/segment {record.segment_id}: zero extent"
        )
    normalization = Normalization(center=center, scale=scale)

    vertex_area = np.bincount(
        record.triangles.ravel(),
        weights=np.repeat(areas / 3.0, 3),
        minlength=record.point_count,
    )
    used_vertices = vertex_area > 0
    facet_used = np.linalg.norm(record.triangle_normals, axis=1) > 0

    return NormalizedSegment(
        mesh_id=record.mesh_id,
        segment_id=record.segment_id,
        normalization=normalization,
        vertices=normalization.points_to_normalized(record.points[used_vertices]),
        vertex_weights=vertex_area[used_vertices] / total_area,
        facet_centroids=normalization.points_to_normalized(centroids[facet_used]),
        facet_normals=record.triangle_normals[facet_used],
        facet_weights=areas[facet_used] / areas[facet_used].sum(),
    )
