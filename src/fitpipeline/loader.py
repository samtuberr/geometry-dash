"""Mesh & segment loader.

Reads `mesh.json` + `segments.json` (or `triangle_segments.csv`) for a given
mesh directory, groups the original triangles by `segment_id`, and exposes
per-segment vertex/triangle arrays as NumPy input for downstream
classification and fitting.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SegmentRecord:
    """Geometry for a single (mesh_id, segment_id) region.

    `points`/`triangles` are compacted to only the vertices referenced by
    this segment: `triangles` indexes into `points`, not the parent mesh's
    vertex array. `triangle_ids` records the original global triangle ids
    (STL facet order) each row in `triangles` came from.
    """

    mesh_id: str
    segment_id: int
    triangle_ids: np.ndarray  # (T,) int64, original triangle indices
    points: np.ndarray  # (V, 3) float64
    triangles: np.ndarray  # (T, 3) int64, indices into `points`
    triangle_areas: np.ndarray  # (T,) float64
    triangle_normals: np.ndarray  # (T, 3) float64, unit vectors (or zero for degenerate triangles)

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def total_area(self) -> float:
        return float(self.triangle_areas.sum())


@dataclass(frozen=True)
class MeshData:
    """A full mesh plus its per-segment breakdown."""

    mesh_id: str
    vertices: np.ndarray  # (N, 3) float64
    triangles: np.ndarray  # (M, 3) int64
    segment_ids: np.ndarray  # (M,) int64, one entry per triangle
    segments: dict[int, SegmentRecord]


def _load_mesh_json(mesh_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = json.loads((mesh_dir / "mesh.json").read_text())
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    triangles = np.asarray(mesh["triangles"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{mesh_dir}/mesh.json: vertices must be (N, 3), got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"{mesh_dir}/mesh.json: triangles must be (M, 3), got {triangles.shape}")
    return vertices, triangles


def _load_segment_ids(mesh_dir: Path, triangle_count: int) -> np.ndarray:
    """Prefer segments.json; fall back to triangle_segments.csv."""
    segments_json = mesh_dir / "segments.json"
    csv_path = mesh_dir / "triangle_segments.csv"

    if segments_json.exists():
        data = json.loads(segments_json.read_text())
        segment_ids = np.asarray(data["triangle_segment_ids"], dtype=np.int64)
    elif csv_path.exists():
        rows = sorted(
            (
                (int(row["triangle_id"]), int(row["segment_id"]))
                for row in csv.DictReader(csv_path.open())
            ),
            key=lambda pair: pair[0],
        )
        segment_ids = np.asarray([segment_id for _, segment_id in rows], dtype=np.int64)
    else:
        raise FileNotFoundError(
            f"{mesh_dir}: neither segments.json nor triangle_segments.csv found"
        )

    if segment_ids.shape[0] != triangle_count:
        raise ValueError(
            f"{mesh_dir}: segment label count ({segment_ids.shape[0]}) does not match "
            f"triangle count ({triangle_count})"
        )
    return segment_ids


def _triangle_geometry(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle area and unit normal. Degenerate triangles get a zero normal."""
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    cross = np.cross(b - a, c - a)
    lengths = np.linalg.norm(cross, axis=1)
    areas = lengths / 2.0
    normals = np.zeros_like(cross)
    nonzero = lengths > 0
    normals[nonzero] = cross[nonzero] / lengths[nonzero, None]
    return areas, normals


def _compact_segment(
    vertices: np.ndarray, triangle_ids: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reindex a subset of triangles onto only the vertices they use."""
    global_rows = triangles[triangle_ids]
    unique_vertex_ids, inverse = np.unique(global_rows.ravel(), return_inverse=True)
    points = vertices[unique_vertex_ids]
    local_triangles = inverse.reshape(global_rows.shape)
    return points, local_triangles


def build_segments(
    mesh_id: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    segment_ids: np.ndarray,
) -> dict[int, SegmentRecord]:
    """Group triangles by segment_id and build a SegmentRecord for each."""
    segments: dict[int, SegmentRecord] = {}
    for segment_id in np.unique(segment_ids):
        triangle_id_arr = np.flatnonzero(segment_ids == segment_id)
        points, local_triangles = _compact_segment(vertices, triangle_id_arr, triangles)
        areas, normals = _triangle_geometry(points, local_triangles)
        segments[int(segment_id)] = SegmentRecord(
            mesh_id=mesh_id,
            segment_id=int(segment_id),
            triangle_ids=triangle_id_arr,
            points=points,
            triangles=local_triangles,
            triangle_areas=areas,
            triangle_normals=normals,
        )
    return segments


def load_mesh_segments(data_dir: Path | str, mesh_id: str) -> MeshData:
    """Load one mesh directory (e.g. `data_dir/mesh_01`) into a MeshData."""
    mesh_dir = Path(data_dir) / mesh_id
    vertices, triangles = _load_mesh_json(mesh_dir)
    segment_ids = _load_segment_ids(mesh_dir, triangle_count=triangles.shape[0])
    segments = build_segments(mesh_id, vertices, triangles, segment_ids)
    return MeshData(
        mesh_id=mesh_id,
        vertices=vertices,
        triangles=triangles,
        segment_ids=segment_ids,
        segments=segments,
    )


def iter_mesh_ids(data_dir: Path | str) -> list[str]:
    """Sorted list of mesh_id directory names under data_dir (e.g. 'mesh_01')."""
    data_dir = Path(data_dir)
    return sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and (p / "mesh.json").exists()
    )


def load_all_meshes(data_dir: Path | str) -> dict[str, MeshData]:
    """Load every mesh_XX directory under data_dir."""
    return {mesh_id: load_mesh_segments(data_dir, mesh_id) for mesh_id in iter_mesh_ids(data_dir)}
