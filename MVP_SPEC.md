# MVP Spec: Segment Classifier & Primitive Fitter

## User Story
As a take-home task reviewer, I want to run one command over the supplied meshes so that I get a `predictions.json` classifying every segment (plane / cylinder / cone / sphere / torus / unresolved) with fitted parameters and a short residual-based justification for each decision.

## Core Features (MVP only — max 5)
1. **Mesh & segment loader**: reads `mesh.json` + `segments.json` (or the CSV) for a given `mesh_id`, groups triangles by `segment_id`, and exposes per-segment vertex/triangle arrays as NumPy input. Input: `data/mesh_XX/`. Output: in-memory segment records (points, triangles, areas, normals).
2. **Classifier**: for each segment, decides `primitive` vs `unresolved`, and if primitive, which of the five supported types. Input: segment geometry (points, normals, areas). Output: `kind`, `primitive_type`, `confidence`, `reason`.
3. **Primitive fitter**: runs SciPy `least_squares` (with NumPy-derived initial guesses, CGAL-assisted guesses optional) for the classified type and returns parameters in the schema's suggested convention (`point`/`normal`, `axis_point`/`axis_direction`/`radius`, etc.). Input: segment geometry + classified type. Output: `fit.status`, `fit.parameters`, `fit.notes`, residual summary.
4. **predictions.json writer**: assembles one entry per `(mesh_id, segment_id)` in the required schema and writes a single valid JSON file. Input: all classified + fitted segments across all 10 meshes. Output: `predictions.json`.
5. **CLI entry point**: one command (`python -m fitpipeline data/ predictions.json`) that runs steps 1–4 over all meshes end-to-end, deterministically, with no network or LLM calls at runtime.

## Out of Scope (explicitly)
- Any UI beyond an optional local Plotly HTML report (not a deliverable requirement per the assignment)
- CAD reconstruction: BRep generation, surface trimming, boundary recovery, watertight solids, STEP/IGES export
- Resegmentation, merging, or splitting of supplied segments
- Arbitrary quadric fitting (only circular cylinder/cone/sphere/torus are supported)
- GPU acceleration or ML model training
- Multi-user access, auth, hosting, or persistence beyond local files
- Perfect or globally optimal fits — imperfect/failed fits are acceptable if explained

## Data Sources
- **Mesh geometry**: `data/mesh_XX/mesh.json` — vertices + triangles, original STL facet order preserved. Local files, no API key, static (fixed for the exercise).
- **Segment labels**: `data/mesh_XX/segments.json` (`triangle_segment_ids`) or `triangle_segments.csv` — one segment ID per original triangle. Same refresh cadence as mesh.json (static).
- **Manifest**: `assignment/manifest.json` — mesh/segment/triangle counts, used for sanity-checking the loader against expected counts (10 meshes, 19,860 total original triangles, 43 total segments).
- **Optional cross-check**: `data/mesh_XX/segmented.ply` — ASCII PLY with per-face segment IDs and display colors, useful for visual inspection only, not required for pipeline logic.

## Tech Stack
- **Runtime**: Python 3.12 + `uv` for environment/dependency management
- **Core numerics**: NumPy (areas, normals, PCA-based initial guesses) + SciPy `optimize.least_squares` (bounded, robust-loss refinement for all five primitives)
- **Optional initializer**: CGAL Efficient RANSAC bindings, time-boxed to 30–45 minutes, only for hard cone/torus initial guesses (separate Python 3.12 venv — CGAL wheels don't support the workspace's default 3.14)
- **Inspection (optional, non-deliverable)**: Plotly, standalone offline HTML export
- **Testing**: pytest — known-geometry fits, partial-surface handling, rotation/translation invariance, rejection of degenerate/insufficient input
- **No auth, no frontend, no hosted backend** — this is a local batch CLI tool, not a service

## Success Criteria
- Running one command processes all 10 meshes and all 43 segments without crashing
- `predictions.json` contains exactly one entry per `(mesh_id, segment_id)`, matching the required schema (`kind`, `primitive_type`, `confidence`, `reason`, `fit.status`, `fit.parameters`, `fit.notes`)
- Every segment classified as a primitive has a fitting attempt recorded (`fitted`, `partial`, or `failed` — never `not_applicable`)
- Fitted parameters are geometrically plausible: axes/normals are unit vectors, radii are positive, positions lie near the segment's bounding region
- At least one pytest check exists for poor-fit rejection and for stability under rotation/translation of a known primitive
- A short written note accompanies the output covering method, fitting objective, parameter conventions, acceptance criteria, confidence meaning, runtime, and known limitations
- Zero LLM/network calls occur during the pipeline's execution (AI is used only to build the code, not to run it)

---

*Generated as part of the take-home task planning phase; see `assignment/README.md` and `assignment/DATA_FORMAT.md` for the authoritative spec this MVP implements.*
