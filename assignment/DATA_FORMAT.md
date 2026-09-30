# Data format

Coordinates use a right-handed Cartesian frame and arbitrary consistent length units within each mesh. JSON is UTF-8. All indices are zero based.

## Original triangle identity

`triangle_id = i` is the i-th facet record in binary `mesh.stl`. It is also row i of `mesh.json["triangles"]`, entry i of `segments.json["triangle_segment_ids"]`, row i after the CSV header, and face i in `segmented.ply`.

Every original triangle has exactly one segment ID. No faces were removed, reordered, subdivided or remeshed. JSON vertices with exactly equal coordinates are deduplicated while preserving every triangle's original corner positions and order. Some STL importers reorder triangles automatically; disable that behavior or use the supplied JSON directly.

## Schemas

`mesh.json`:

```json
{"vertices": [[0,0,0], [1,0,0], [0,1,0]], "triangles": [[0,1,2]]}
```

`segments.json`:

```json
{"triangle_segment_ids": [0]}
```

`triangle_segments.csv`:

```csv
triangle_id,segment_id
0,0
```

These tiny examples illustrate the format only. The segmentation array length equals the number of original triangles. Segment IDs are consecutive nonnegative integers local to each mesh; their numeric values do not encode classes. A segment comprises all triangles sharing its ID.

Minimal Python ingestion, using only the standard library:

```python
import json
from pathlib import Path
folder = Path("data/mesh_01")
mesh = json.loads((folder / "mesh.json").read_text())
labels = json.loads((folder / "segments.json").read_text())["triangle_segment_ids"]
segments = {}
for triangle_id, segment_id in enumerate(labels):
    segments.setdefault(segment_id, []).append(triangle_id)
# For a triangle: points = [mesh["vertices"][i] for i in mesh["triangles"][triangle_id]]
```

`segmented.ply` is ASCII PLY with the same vertex and triangle order as JSON. Each face carries `vertex_indices`, RGB color and integer `segment_id`. Colors distinguish regions only and have no class meaning. Some PLY viewers ignore per-face colors; the CSV/JSON labels are unambiguous.

`manifest.json` gives relative directories, vertex/triangle/segment counts and per-segment triangle counts.

## Supplied segmentation and provenance

Segments were formed by connected-component growth across manifold edges with adjacent face-normal angles below 30 degrees. Sharp edges, boundary edges and non-manifold edges block growth. Exactly coincident STL vertices were welded for adjacency only. All supplied segments were checked for edge connectivity and every original triangle was checked for matching identity across STL, JSON, CSV and PLY.

All meshes are procedural, non-customer fixtures: seven existing internally generated demonstration meshes and three procedural additions, including one deformation of a procedural fixture. No customer scans, customer CAD or downloaded commercial models are included. These are clean synthetic tessellations, not noisy production scans. Source construction names and primitive labels are intentionally omitted from the candidate inputs.

## Stage 2 output: primitive parameters

Include a `fit` object in each prediction as described in `README.md`. Report parameters in the input mesh's coordinate frame and length units. If you normalize coordinates during fitting, transform the final parameters back to the input frame. Direction vectors should be normalized. Document any alternative representation clearly; the following field names are suggested, not mandatory.

| Primitive | Suggested parameters | Meaning |
| --- | --- | --- |
| Plane | `point`, `normal` | A point on the plane and its unit normal. |
| Cylinder | `axis_point`, `axis_direction`, `radius` | Any point on the axis, its unit direction, and a positive radius. |
| Cone | `apex`, `axis_direction`, `half_angle_radians` | Apex, unit axis directed toward the fitted cone portion, and the angle between the axis and the cone surface, between 0 and pi/2. An apex outside a truncated segment is valid. |
| Sphere | `center`, `radius` | Center and positive radius. |
| Torus | `center`, `axis_direction`, `major_radius`, `minor_radius` | Center, unit symmetry axis, center-to-tube-centerline radius, and tube radius. |

Points and directions are three-number arrays. Scalar parameters must be finite. For a partial fit, return only usable estimates and explain what is missing. Use JSON `null` for `parameters` if the attempt produced no usable estimate; do not emit NaN or Infinity. No finite height, trimming interval, boundary curve, or surface tessellation is required.

Illustrative fit object for a cylinder (not a dataset answer):

```json
{
  "status": "fitted",
  "parameters": {
    "axis_point": [0.0, 0.0, 0.0],
    "axis_direction": [0.0, 0.0, 1.0],
    "radius": 2.0
  },
  "notes": "Describe the fitting method and evidence that these parameters follow the segment."
}
```

`fitted` means you obtained an estimate you consider usable, not an exact or globally optimal solution. `partial` means some useful estimates were obtained but the fit is incomplete or uncertain. `failed` records an attempted fit without a usable result. `not_applicable` means no primitive was selected and no fitting attempt was made. Optional residual fields should identify their units, sampling and aggregation method. Equivalent parameterizations, such as different points on the same cylinder axis or opposite plane normal signs, can represent the same surface.
