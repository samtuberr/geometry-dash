# Take-home task: classify segments and fit primitive surfaces

You receive 10 non-customer, procedurally generated meshes of simple to medium complexity. Their 19,860 original triangles are already partitioned into 43 segments. Your task has two stages: **classify each segment**, then **attempt to fit the identified primitive and recover its parameters**.

## Stage 1: classify segments

For every supplied segment:

1. Decide whether its surface is adequately explained by a supported primitive or should be marked `unresolved` (freeform, unsupported geometry, poor fit, or insufficient evidence).
2. If it is a primitive, identify its type: **plane, cylinder, cone, sphere, or torus**.

Classify the underlying surface approximated by the triangles, not each individual planar triangle. A segment can cover only part of a primitive. Supported curved primitives are circular cylinders, circular cones, spheres and circular tori; arbitrary quadrics are not included.

Segmentation is already supplied. Do not split, merge or resegment the regions. No CAD reconstruction, BRep generation, GPU implementation or machine-learning training is required. Use geometry rather than filenames, segment IDs or hard-coded answers. Numerical fitting and other geometric approaches are welcome.

## Stage 2: attempt primitive fitting

For every segment identified as a primitive, attempt to fit the best instance of that primitive type to its geometry and report the fitted parameters. Choose and explain a fitting objective appropriate to your method; a globally optimal solution or a successful fit for every segment is not required. Classification and fitting may share code or run together internally.

The goal is a reasonable fitting attempt with parameters that make geometric sense: positions and axes should align with the segment, sizes should be plausible, and the estimated surface should follow the supplied geometry. Report a short fit assessment, ideally with a residual or another simple geometric check. Document the samples, units and error measure if you report one.

An imperfect or failed fit is acceptable when you show the attempt and explain the limitation. Return partial estimates when useful, and mark failures explicitly rather than inventing parameters. If fitting reveals that the primitive classification is unsupported, you may revise the final classification to `unresolved` and explain the attempted fit.

**A complete mesh-to-CAD conversion is not required.** There is no requirement to trim surfaces, recover face boundaries, stitch surfaces, build a watertight solid, or export STEP/BRep. We are interested in your approach and sensible primitive parameters, not a finished conversion pipeline.

## Tools and libraries

You may use any existing packages, libraries, frameworks, solvers or other tools that help you complete the task. You may also create any visualization or interactive inspection tool you find useful for checking segments and fits. A UI is optional and is not a deliverable requirement. Document the dependencies, how to run your solution, and which parts you implemented versus reused.

## Inputs

Each `data/mesh_XX/` directory contains:

- `mesh.stl`: original triangle mesh.
- `mesh.json`: indexed vertices and triangles, preserving original STL facet order.
- `segments.json`: one `segment_id` per original triangle, as `triangle_segment_ids`.
- `triangle_segments.csv`: the same mapping with explicit `triangle_id,segment_id` columns.
- `segmented.ply`: the mesh with per-face segment IDs and optional display colors.

Use whichever encoding is convenient. `manifest.json` lists counts and paths. See `DATA_FORMAT.md` for exact indexing and schemas. All 10 meshes are procedural fixtures; none is customer data.

## Required output

Write `predictions.json` with exactly one entry for each `(mesh_id, segment_id)`:

```json
{
  "predictions": [
    {
      "mesh_id": "mesh_01",
      "segment_id": 0,
      "kind": "unresolved",
      "primitive_type": null,
      "confidence": 0.4,
      "reason": "Illustrative schema only; replace with geometric evidence.",
      "fit": {
        "status": "not_applicable",
        "parameters": null,
        "notes": "No primitive selected in this illustrative example."
      }
    }
  ]
}
```

This is a schema example, not an expected answer. `kind` must be `primitive` or `unresolved`. For primitives, `primitive_type` must be `plane`, `cylinder`, `cone`, `sphere`, or `torus`; otherwise it must be JSON `null`. Confidence is a finite number in [0,1]. Include a short reason for each decision.

Include a `fit` object in each entry, with `status`, `parameters` and `notes`. Status is `fitted`, `partial`, `failed`, or `not_applicable`. For segments classified as primitives, a fitting attempt is required: use `fitted`, `partial`, or `failed`. Use `not_applicable` only when no primitive was selected. If you attempted a fit and then revised the classification to `unresolved`, retain the attempt status and identify the attempted type in the notes.

`parameters` contains the best available estimates, or JSON `null` if no usable estimate was obtained. `notes` briefly explains the result or failure. See `DATA_FORMAT.md` for suggested parameter conventions. Residual summaries are encouraged but optional.

Use scale-appropriate tolerances and account for tessellation error. Prefer an explained refusal to forcing a poor primitive fit. State how you handle ambiguity, uneven triangle sizes, partial surfaces and numerical failures.

## Deliverables

- Source code, dependency instructions, and one command that reads the dataset and writes predictions.
- Your `predictions.json` for all supplied segments, including fitting attempts and available primitive parameters.
- A short note explaining the classification and fitting methods, fitting objective, parameter conventions, fit acceptance criteria, confidence meaning, approximate runtime, limitations, and time spent.
- A few focused checks of properties you consider important, such as poor-fit rejection or stability under changes in scale or orientation.

Suggested time budget: **6–8 hours**. Identify unfinished work rather than exceeding the timebox to build a production system. We will assess geometric reasoning, classification and rejection quality, the fitting attempt and whether its parameters make sense, numerical robustness, reproducibility and clarity. A complete conversion or perfect fits are not expected. These ten clean synthetic meshes are a small exercise, not a statistically representative production benchmark.
