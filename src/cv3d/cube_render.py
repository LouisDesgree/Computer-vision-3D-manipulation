from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np

from .palette import IOS_BLUE, IOS_BLUE_SOFT, IOS_KNOB
BASE_VERTICES = [
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
]

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

FACES = [
    ([0, 1, 2, 3], (96, 96, 104)),
    ([4, 5, 6, 7], (108, 108, 118)),
    ([0, 1, 5, 4], (90, 90, 98)),
    ([2, 3, 7, 6], (118, 118, 128)),
    ([1, 2, 6, 5], (100, 100, 108)),
    ([0, 3, 7, 4], (112, 112, 122)),
]


def _rotate_point(
    point: Tuple[float, float, float],
    yaw: float,
    pitch: float,
) -> Tuple[float, float, float]:
    x, y, z = point
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    xz = x * cos_y + z * sin_y
    zz = -x * sin_y + z * cos_y
    yz = y * cos_p - zz * sin_p
    zz = y * sin_p + zz * cos_p
    return xz, yz, zz


def _normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = vec
    length = math.sqrt(x * x + y * y + z * z) or 1.0
    return x / length, y / length, z / length


def _face_normal(
    v0: Tuple[float, float, float],
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    ax, ay, az = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    bx, by, bz = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    return _normalize((nx, ny, nz))


def _reflect(light: Tuple[float, float, float], normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    dot_ln = light[0] * normal[0] + light[1] * normal[1] + light[2] * normal[2]
    rx = 2.0 * dot_ln * normal[0] - light[0]
    ry = 2.0 * dot_ln * normal[1] - light[1]
    rz = 2.0 * dot_ln * normal[2] - light[2]
    return _normalize((rx, ry, rz))


def _to_poly(points: List[Tuple[int, int]]):
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


class CubeRenderer:
    def __init__(self, size: float, distance: float) -> None:
        self.size = size
        self.distance = distance

    def project(
        self,
        center: Tuple[int, int],
        yaw_deg: float,
        pitch_deg: float,
        focal_length: float,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float, float]]]:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        cx, cy = center
        sx, sy, sz = scale
        projected: List[Tuple[int, int]] = []
        rotated: List[Tuple[float, float, float]] = []
        for vx, vy, vz in BASE_VERTICES:
            point = (vx * self.size * sx, vy * self.size * sy, vz * self.size * sz)
            rx, ry, rz = _rotate_point(point, yaw, pitch)
            rz += self.distance
            rotated.append((rx, ry, rz))
            if rz <= 0.01:
                rz = 0.01
            u = int(cx + (focal_length * rx / rz))
            v = int(cy + (focal_length * ry / rz))
            projected.append((u, v))
        return projected, rotated

    def bounds(self, projected: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in projected]
        ys = [p[1] for p in projected]
        return min(xs), min(ys), max(xs), max(ys)

    def draw(
        self,
        frame,
        projected: List[Tuple[int, int]],
        rotated: List[Tuple[float, float, float]],
        highlight: bool,
    ) -> None:
        light_dir = _normalize((0.6, 0.9, 0.4))
        view_dir = (0.0, 0.0, 1.0)
        accent = IOS_BLUE_SOFT
        face_depths = []
        for face_indices, base_color in FACES:
            verts = [rotated[idx] for idx in face_indices]
            avg_z = sum(v[2] for v in verts) / len(verts)
            normal = _face_normal(verts[0], verts[1], verts[2])
            diffuse = max(
                normal[0] * light_dir[0] + normal[1] * light_dir[1] + normal[2] * light_dir[2],
                0.0,
            )
            rim = 1.0 - max(
                normal[0] * view_dir[0] + normal[1] * view_dir[1] + normal[2] * view_dir[2],
                0.0,
            )
            rim = rim ** 2.1
            reflect = _reflect(light_dir, normal)
            spec = max(
                reflect[0] * view_dir[0] + reflect[1] * view_dir[1] + reflect[2] * view_dir[2],
                0.0,
            )
            spec = spec ** 10

            base_scale = 0.28 + 0.72 * diffuse
            if highlight:
                base_scale = min(1.25, base_scale + 0.08)
                rim = min(1.0, rim * 1.2)
                spec = min(1.0, spec * 1.3)

            shade = []
            for channel, base in enumerate(base_color):
                value = (
                    base * base_scale
                    + accent[channel] * rim * 0.35
                    + 255.0 * spec * 0.2
                )
                shade.append(int(max(0, min(255, value))))
            shade = tuple(shade)
            face_depths.append((avg_z, face_indices, shade))

        for _avg_z, face_indices, shade in sorted(
            face_depths,
            key=lambda item: item[0],
            reverse=True,
        ):
            pts = [projected[idx] for idx in face_indices]
            cv2.fillConvexPoly(frame, _to_poly(pts), shade)

        edge_outer = IOS_BLUE_SOFT
        edge_inner = IOS_BLUE
        outer_thickness = 3
        inner_thickness = 1
        if highlight:
            edge_outer = IOS_BLUE
            edge_inner = IOS_KNOB
            outer_thickness = 4
            inner_thickness = 2
        for start, end in EDGES:
            cv2.line(frame, projected[start], projected[end], edge_outer, outer_thickness)
            cv2.line(frame, projected[start], projected[end], edge_inner, inner_thickness)
