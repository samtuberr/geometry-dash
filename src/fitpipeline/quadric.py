"""General-quadric diagnostic for segments no supported primitive explains.

The supported primitives are special quadrics (plane, sphere, circular cylinder
and cone) plus the torus, which is quartic. When all of them are rejected, a fit
of the general quadric

    x^T A x + b . x + d = 0

tells the two remaining cases apart: an unsupported quadric such as an
ellipsoid or a hyperboloid (vertices on it to tessellation precision), or a
freeform surface (no quadric follows it). The result only enriches the reason
of an `unresolved` decision; it never turns a segment into a primitive.

The fit is the smallest right singular vector of the area-weighted design
matrix in the segment's normalized frame, and it is scored by the first-order
(Sampson) geometric distance |Q(x)| / |grad Q(x)|, so the score is in the same
relative units as the primitive residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fitpipeline.geometry import NormalizedSegment

# Ten coefficients, so ask for a few spare vertices before trusting a fit.
MIN_VERTICES = 13
# Eigenvalues of A below this fraction of the largest count as zero.
_SINGULAR_TOL = 1e-6


@dataclass(frozen=True)
class QuadricFit:
    """Best general quadric through a segment's vertices, described in input units."""

    relative_rms: float  # area-weighted RMS Sampson distance / segment RMS extent
    kind: str  # "ellipsoid", "hyperboloid of one sheet", ..., or "non-central quadric"
    center: np.ndarray | None  # input frame; None for non-central quadrics
    semi_axes: np.ndarray | None  # ellipsoid only, input units, descending

    def describe(self) -> str:
        if self.kind == "ellipsoid":
            axes = ", ".join(f"{a:.4g}" for a in self.semi_axes)
            centre = ", ".join(f"{c:.4g}" for c in self.center)
            return f"an ellipsoid with semi-axes {axes} centred at ({centre})"
        return f"a {self.kind}"


def fit_quadric(segment: NormalizedSegment) -> QuadricFit | None:
    """Fit a general quadric to the vertices, or `None` if there are too few of them."""
    x, w = segment.vertices, segment.vertex_weights
    if len(x) < MIN_VERTICES:
        return None
    px, py, pz = x.T
    ones = np.ones_like(px)
    design = np.column_stack(
        [px * px, py * py, pz * pz, 2 * px * py, 2 * px * pz, 2 * py * pz,
         2 * px, 2 * py, 2 * pz, ones]
    )
    _, _, right_vectors = np.linalg.svd(design * np.sqrt(w)[:, None], full_matrices=False)
    a11, a22, a33, a12, a13, a23, b1, b2, b3, d = right_vectors[-1]
    matrix = np.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]])
    linear = np.array([b1, b2, b3])

    values = np.einsum("ij,jk,ik->i", x, matrix, x) + 2 * x @ linear + d
    gradients = 2 * (x @ matrix + linear)
    gradient_norms = np.linalg.norm(gradients, axis=1)
    sampson = np.divide(
        np.abs(values), gradient_norms, out=np.full_like(values, np.inf), where=gradient_norms > 0
    )
    relative_rms = float(np.sqrt(w @ np.minimum(sampson, 1e3) ** 2))

    scale = segment.normalization.scale
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(np.abs(eigenvalues)) <= _SINGULAR_TOL * np.max(np.abs(eigenvalues)):
        kind = "non-central quadric (paraboloid or elliptic/hyperbolic cylinder)"
        return QuadricFit(relative_rms, kind, None, None)

    # Centred form (x - c)^T A (x - c) = k with c = -A^-1 b and k = c^T A c - d.
    center = -np.linalg.solve(matrix, linear)
    level = float(center @ matrix @ center - d)
    center_input = segment.normalization.points_from_normalized(center)
    # Float32 input carries ~1e-7 relative noise; do not report it as an offset.
    center_input[np.abs(center_input) < 1e-7 * scale] = 0.0
    positive = int(np.sum(eigenvalues * level > 0))
    if level == 0:
        kind = "elliptic cone"
    else:
        kind = {
            3: "ellipsoid",
            2: "hyperboloid of one sheet",
            1: "hyperboloid of two sheets",
            0: "imaginary quadric",
        }[positive]
    semi_axes = None
    if kind == "ellipsoid":
        semi_axes = np.sort(np.sqrt(level / eigenvalues))[::-1] * scale
    return QuadricFit(relative_rms, kind, center_input, semi_axes)
