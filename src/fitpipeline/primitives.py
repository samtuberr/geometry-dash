"""Primitive parameter conventions, point-to-surface distances and surface normals.

Parameter names follow the suggested schema in `assignment/DATA_FORMAT.md`.
Every distance/normal function takes the parameters of its primitive as keyword
arguments, so `surface_distance(type, parameters, points)` can dispatch with
`**parameters`. Distances are unsigned and measured to the infinite surface;
normals are unit vectors whose sign is arbitrary.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from fitpipeline.geometry import Normalization

PRIMITIVE_TYPES = ("plane", "cylinder", "cone", "sphere", "torus")

Parameters = dict[str, "np.ndarray | float"]

# Which parameters are positions and which are lengths; directions and angles
# are unchanged by a similarity transform without rotation.
POINT_PARAMETERS = {
    "plane": ("point",),
    "cylinder": ("axis_point",),
    "cone": ("apex",),
    "sphere": ("center",),
    "torus": ("center",),
}
LENGTH_PARAMETERS = {
    "plane": (),
    "cylinder": ("radius",),
    "cone": (),
    "sphere": ("radius",),
    "torus": ("major_radius", "minor_radius"),
}


def parameters_from_normalized(
    primitive_type: str, parameters: Parameters, normalization: Normalization
) -> Parameters:
    """Map parameters fitted in the normalized frame back to the input frame."""
    result = dict(parameters)
    for key in POINT_PARAMETERS[primitive_type]:
        result[key] = normalization.points_from_normalized(parameters[key])
    for key in LENGTH_PARAMETERS[primitive_type]:
        result[key] = normalization.lengths_from_normalized(parameters[key])
    return result


def parameters_to_jsonable(parameters: Parameters) -> dict[str, list[float] | float]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in parameters.items()
    }


def _split_axial_radial(
    points: np.ndarray, origin: np.ndarray, axis_direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Axial coordinate (N,) and radial offset vectors (N, 3) about an axis."""
    offsets = points - origin
    axial = offsets @ axis_direction
    return axial, offsets - axial[:, None] * axis_direction


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)


def distance_to_plane(points, point, normal):
    return np.abs((points - point) @ normal)


def distance_to_sphere(points, center, radius):
    return np.abs(np.linalg.norm(points - center, axis=1) - radius)


def distance_to_cylinder(points, axis_point, axis_direction, radius):
    _, radial = _split_axial_radial(points, axis_point, axis_direction)
    return np.abs(np.linalg.norm(radial, axis=1) - radius)


def distance_to_cone(points, apex, axis_direction, half_angle_radians):
    """Distance to the nappe opening from `apex` along `axis_direction`."""
    axial, radial_vectors = _split_axial_radial(points, apex, axis_direction)
    radial = np.linalg.norm(radial_vectors, axis=1)
    cos_a, sin_a = np.cos(half_angle_radians), np.sin(half_angle_radians)
    along_generator = axial * cos_a + radial * sin_a
    to_line = np.abs(radial * cos_a - axial * sin_a)
    # Behind the apex the nearest surface point is the apex itself.
    return np.where(along_generator >= 0, to_line, np.hypot(axial, radial))


def distance_to_torus(points, center, axis_direction, major_radius, minor_radius):
    axial, radial_vectors = _split_axial_radial(points, center, axis_direction)
    radial = np.linalg.norm(radial_vectors, axis=1)
    return np.abs(np.hypot(radial - major_radius, axial) - minor_radius)


def normal_of_plane(points, point, normal):
    return np.tile(normal, (len(points), 1))


def normal_of_sphere(points, center, radius):
    return _unit(points - center)


def normal_of_cylinder(points, axis_point, axis_direction, radius):
    _, radial = _split_axial_radial(points, axis_point, axis_direction)
    return _unit(radial)


def normal_of_cone(points, apex, axis_direction, half_angle_radians):
    _, radial = _split_axial_radial(points, apex, axis_direction)
    # Perpendicular to the generator cos(a)*axis + sin(a)*radial, in the same plane.
    return _unit(
        np.sin(half_angle_radians) * axis_direction
        - np.cos(half_angle_radians) * _unit(radial)
    )


def normal_of_torus(points, center, axis_direction, major_radius, minor_radius):
    axial, radial = _split_axial_radial(points, center, axis_direction)
    to_tube_center = major_radius * _unit(radial)
    return _unit(radial - to_tube_center + axial[:, None] * axis_direction)


_DISTANCE_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "plane": distance_to_plane,
    "cylinder": distance_to_cylinder,
    "cone": distance_to_cone,
    "sphere": distance_to_sphere,
    "torus": distance_to_torus,
}


def surface_distance(
    primitive_type: str, parameters: Parameters, points: np.ndarray
) -> np.ndarray:
    return _DISTANCE_FUNCTIONS[primitive_type](points, **parameters)


_NORMAL_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "plane": normal_of_plane,
    "cylinder": normal_of_cylinder,
    "cone": normal_of_cone,
    "sphere": normal_of_sphere,
    "torus": normal_of_torus,
}


def surface_normal(
    primitive_type: str, parameters: Parameters, points: np.ndarray
) -> np.ndarray:
    return _NORMAL_FUNCTIONS[primitive_type](points, **parameters)
