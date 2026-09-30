"""Tessellated partial primitives with known parameters, for testing."""

from __future__ import annotations

import numpy as np

from fitpipeline.loader import SegmentRecord, build_segments


def random_rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else q[:, ::-1] * np.array([1, 1, -1])


class Pose:
    """Rotation, uniform scale and translation applied to a shape and its parameters."""

    def __init__(self, rotation=None, scale=1.0, translation=(0.0, 0.0, 0.0)):
        self.rotation = np.eye(3) if rotation is None else rotation
        self.scale = scale
        self.translation = np.asarray(translation, dtype=np.float64)

    def point(self, p):
        return self.scale * (np.asarray(p, dtype=np.float64) @ self.rotation.T) + self.translation

    def direction(self, d):
        return np.asarray(d, dtype=np.float64) @ self.rotation.T


IDENTITY = Pose()


def grid_segment(surface, u_values, v_values, pose: Pose = IDENTITY) -> SegmentRecord:
    """Triangulate `surface(u, v) -> (x, y, z)` sampled on a (u, v) grid, then apply `pose`."""
    u, v = np.meshgrid(u_values, v_values, indexing="ij")
    points = pose.point(np.stack(surface(u, v), axis=-1).reshape(-1, 3))
    nv = len(v_values)
    index = np.arange(len(u_values) * nv).reshape(len(u_values), nv)
    a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)])
    segment_ids = np.zeros(len(triangles), dtype=np.int64)
    return build_segments("synthetic", points, triangles, segment_ids)[0]


def uneven(start, stop, count):
    """Geometrically graded samples: triangle sizes vary by ~10x across the grid."""
    t = np.linspace(0.0, 1.0, count) ** 2
    return start + (stop - start) * t


def plane(pose=IDENTITY):
    record = grid_segment(
        lambda u, v: (u, v, 0 * u), uneven(0, 4, 6), np.linspace(-1, 2, 5), pose
    )
    return record, {"normal": pose.direction([0, 0, 1])}


def sphere(pose=IDENTITY, center=(0.5, -0.3, 0.2), radius=2.0):
    c = np.asarray(center)
    record = grid_segment(
        lambda p, t: (
            c[0] + radius * np.sin(p) * np.cos(t),
            c[1] + radius * np.sin(p) * np.sin(t),
            c[2] + radius * np.cos(p) + 0 * t,
        ),
        uneven(0.6, 2.2, 12),
        np.linspace(0.0, 4.0, 24),
        pose,
    )
    return record, {"center": pose.point(c), "radius": pose.scale * radius}


def cylinder(pose=IDENTITY, radius=1.5):
    record = grid_segment(
        lambda t, z: (radius * np.cos(t), radius * np.sin(t), z),
        np.linspace(0.2, 2.5, 20),
        uneven(0.0, 4.0, 5),
        pose,
    )
    return record, {
        "axis_direction": pose.direction([0, 0, 1]),
        "axis_point": pose.point([0, 0, 0]),
        "radius": pose.scale * radius,
    }


def cone(pose=IDENTITY, half_angle=0.5):
    record = grid_segment(
        lambda t, s: (
            s * np.sin(half_angle) * np.cos(t),
            s * np.sin(half_angle) * np.sin(t),
            s * np.cos(half_angle),
        ),
        np.linspace(0.0, 3.0, 24),
        uneven(1.0, 3.0, 6),
        pose,
    )
    return record, {
        "apex": pose.point([0, 0, 0]),
        "axis_direction": pose.direction([0, 0, 1]),
        "half_angle_radians": half_angle,
    }


def torus(pose=IDENTITY, major=3.0, minor=1.0):
    record = grid_segment(
        lambda t, p: (
            (major + minor * np.cos(p)) * np.cos(t),
            (major + minor * np.cos(p)) * np.sin(t),
            minor * np.sin(p),
        ),
        np.linspace(0.0, 3.6, 60),
        np.linspace(0.0, 4.4, 40),
        pose,
    )
    return record, {
        "center": pose.point([0, 0, 0]),
        "axis_direction": pose.direction([0, 0, 1]),
        "major_radius": pose.scale * major,
        "minor_radius": pose.scale * minor,
    }
