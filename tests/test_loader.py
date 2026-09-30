import json
from pathlib import Path

import numpy as np
import pytest

from fitpipeline.loader import iter_mesh_ids, load_all_meshes, load_mesh_segments

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1] / "assignment"
DATA_DIR = ASSIGNMENT_DIR / "data"
MANIFEST = json.loads((ASSIGNMENT_DIR / "manifest.json").read_text())


def test_iter_mesh_ids_matches_manifest():
    expected = sorted(m["mesh_id"] for m in MANIFEST["meshes"])
    assert iter_mesh_ids(DATA_DIR) == expected


@pytest.mark.parametrize("mesh_info", MANIFEST["meshes"], ids=lambda m: m["mesh_id"])
def test_mesh_matches_manifest_counts(mesh_info):
    mesh = load_mesh_segments(DATA_DIR, mesh_info["mesh_id"])

    assert mesh.vertices.shape == (mesh_info["vertex_count"], 3)
    assert mesh.triangles.shape == (mesh_info["triangle_count"], 3)
    assert len(mesh.segments) == mesh_info["segment_count"]

    expected_triangle_counts = {
        int(segment_id): count
        for segment_id, count in mesh_info["segment_triangle_counts"].items()
    }
    actual_triangle_counts = {
        segment_id: record.triangle_count for segment_id, record in mesh.segments.items()
    }
    assert actual_triangle_counts == expected_triangle_counts


@pytest.mark.parametrize("mesh_info", MANIFEST["meshes"], ids=lambda m: m["mesh_id"])
def test_segment_triangles_partition_original_mesh(mesh_info):
    mesh = load_mesh_segments(DATA_DIR, mesh_info["mesh_id"])

    all_triangle_ids = np.concatenate(
        [record.triangle_ids for record in mesh.segments.values()]
    )
    assert sorted(all_triangle_ids.tolist()) == list(range(mesh.triangles.shape[0]))


@pytest.mark.parametrize("mesh_info", MANIFEST["meshes"], ids=lambda m: m["mesh_id"])
def test_segment_geometry_arrays_are_consistent(mesh_info):
    mesh = load_mesh_segments(DATA_DIR, mesh_info["mesh_id"])

    for record in mesh.segments.values():
        t = record.triangle_count
        assert record.triangles.shape == (t, 3)
        assert record.triangle_areas.shape == (t,)
        assert record.triangle_normals.shape == (t, 3)
        assert record.triangle_ids.shape == (t,)
        # triangle indices must be valid into the compacted points array
        assert record.triangles.max(initial=-1) < record.point_count
        assert record.triangles.min(initial=0) >= 0
        # areas are non-negative; normals are unit length or zero (degenerate)
        assert np.all(record.triangle_areas >= 0)
        norm_lengths = np.linalg.norm(record.triangle_normals, axis=1)
        assert np.all((np.isclose(norm_lengths, 1.0)) | (norm_lengths == 0))

        # a triangle's original vertex positions must be reproducible via the
        # compacted points/triangles arrays and the source mesh
        for local_row, global_id in zip(record.triangles, record.triangle_ids):
            expected = mesh.vertices[mesh.triangles[global_id]]
            actual = record.points[local_row]
            np.testing.assert_array_equal(actual, expected)


def test_load_all_meshes_covers_every_mesh():
    meshes = load_all_meshes(DATA_DIR)
    assert set(meshes.keys()) == {m["mesh_id"] for m in MANIFEST["meshes"]}


def test_missing_segments_json_falls_back_to_csv(tmp_path):
    mesh_dir = tmp_path / "mesh_00"
    mesh_dir.mkdir()
    (mesh_dir / "mesh.json").write_text(
        json.dumps({"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "triangles": [[0, 1, 2]]})
    )
    (mesh_dir / "triangle_segments.csv").write_text("triangle_id,segment_id\n0,0\n")

    mesh = load_mesh_segments(tmp_path, "mesh_00")

    assert len(mesh.segments) == 1
    assert mesh.segments[0].triangle_count == 1


def test_mismatched_segment_label_count_raises(tmp_path):
    mesh_dir = tmp_path / "mesh_00"
    mesh_dir.mkdir()
    (mesh_dir / "mesh.json").write_text(
        json.dumps({"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "triangles": [[0, 1, 2]]})
    )
    (mesh_dir / "segments.json").write_text(json.dumps({"triangle_segment_ids": [0, 1]}))

    with pytest.raises(ValueError, match="does not match"):
        load_mesh_segments(tmp_path, "mesh_00")
