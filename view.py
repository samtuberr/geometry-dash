#!/usr/bin/env python3
"""Inspect one segment: its closed-form seeds per primitive type and the final decision.

Usage (from the repository root):
  uv run python view.py mesh_02 0          # See segment 0 of mesh_02
  uv run python view.py mesh_03 0          # Cone frustum
  uv run python view.py mesh_08 0          # Torus
"""

from pathlib import Path
import sys
import numpy as np

from fitpipeline import classify_segment, load_mesh_segments, propose_hypotheses


def print_hypothesis(h, index):
    """Pretty-print one hypothesis."""
    print(f"\n  [{index}] {h.primitive_type.upper()}")
    print(f"      Vertex residual (RMS):     {h.relative_rms:.2e} (normalized frame)")
    print(f"      Vertex residual (input):   {h.rms_distance:.6f}")
    print(f"      Facet normal angle (RMS):  {np.degrees(h.normal_rms_radians):.2f}°")

    for key, value in sorted(h.parameters.items()):
        if isinstance(value, np.ndarray):
            if len(value) == 3:
                print(f"      {key:20s} = [{value[0]:8.5f}, {value[1]:8.5f}, {value[2]:8.5f}]")
            else:
                print(f"      {key:20s} = {value}")
        else:
            if isinstance(value, float):
                print(f"      {key:20s} = {value:.8f}")
            else:
                print(f"      {key:20s} = {value}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable meshes:")
        data_dir = Path("assignment/data")
        for mesh_dir in sorted(data_dir.glob("mesh_*")):
            mesh_id = mesh_dir.name
            mesh = load_mesh_segments(data_dir, mesh_id)
            segment_ids = sorted(mesh.segments.keys())
            print(f"  {mesh_id}: segments {segment_ids}")
        sys.exit(1)

    mesh_id = sys.argv[1]
    segment_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    data_dir = Path("assignment/data")
    mesh = load_mesh_segments(data_dir, mesh_id)
    record = mesh.segments[segment_id]

    print(f"\n{'='*70}")
    print(f"  {mesh_id}, segment {segment_id}")
    print(f"{'='*70}")
    print(f"  Triangles: {record.triangle_count}")
    print(f"  Vertices:  {record.point_count}")
    print(f"  Total area: {record.total_area:.4f}")
    print(f"  Bounding box (input frame):")
    bmin, bmax = record.points.min(axis=0), record.points.max(axis=0)
    extent = bmax - bmin
    print(f"    min: [{bmin[0]:.3f}, {bmin[1]:.3f}, {bmin[2]:.3f}]")
    print(f"    max: [{bmax[0]:.3f}, {bmax[1]:.3f}, {bmax[2]:.3f}]")
    print(f"    extent: [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")

    hypotheses = propose_hypotheses(record)
    print(f"\n  Initial hypotheses for {len(hypotheses)} primitive type(s):\n")

    for i, h in enumerate(hypotheses):
        print_hypothesis(h, i)

    best = min(hypotheses, key=lambda h: h.relative_rms)
    print(f"\n  ★ Best fit by vertex residual: {best.primitive_type.upper()}")
    print(f"    (RMS: {best.relative_rms:.2e}, normal angle: {np.degrees(best.normal_rms_radians):.1f}°)")
    print("    Seeds only; the decision below uses refined fits and both tolerances.")

    result = classify_segment(record)
    print(f"\n  Final decision: {result.primitive_type or 'unresolved'} (confidence {result.confidence})")
    print(f"    {result.reason}")
    print()


if __name__ == "__main__":
    main()
