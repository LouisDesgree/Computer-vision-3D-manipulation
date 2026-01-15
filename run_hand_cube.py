#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cv3d.hand_tracking import HandTracker


@dataclass
class CubeState:
    yaw: float = 20.0
    pitch: float = -15.0
    yaw_target: float = 20.0
    pitch_target: float = -15.0
    dragging: bool = False
    last_tip: Optional[Tuple[int, int]] = None
    trail: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=32))
    last_hand_time: float = 0.0
    center: Optional[Tuple[int, int]] = None


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
    ([0, 1, 2, 3], (66, 166, 242)),
    ([4, 5, 6, 7], (242, 140, 64)),
    ([0, 1, 5, 4], (90, 190, 120)),
    ([2, 3, 7, 6], (224, 120, 160)),
    ([1, 2, 6, 5], (242, 222, 64)),
    ([0, 3, 7, 4], (90, 110, 242)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay a 3D cube on the camera feed and rotate it by touch."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--auto-camera",
        action="store_true",
        help="Try next camera index if frames are black.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=4,
        help="Highest camera index to probe with --auto-camera.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "any", "avfoundation"),
        default="auto",
        help="Camera backend (macOS default is AVFoundation).",
    )
    parser.add_argument("--flip", action="store_true", help="Mirror the camera view.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to hand_landmarker.task.",
    )
    parser.add_argument(
        "--show-camera",
        action="store_true",
        help="(Deprecated) camera feed is always shown in this demo.",
    )
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--trail-length", type=int, default=32)
    parser.add_argument("--sensitivity", type=float, default=0.32)
    parser.add_argument("--rotation-smoothing", type=float, default=0.18)
    parser.add_argument("--anchor-smoothing", type=float, default=0.14)
    parser.add_argument("--max-anchor-step", type=float, default=40.0)
    parser.add_argument("--cube-size", type=float, default=0.6)
    parser.add_argument("--cube-distance", type=float, default=5.8)
    parser.add_argument(
        "--cube-anchor",
        choices=("hand", "center"),
        default="hand",
        help="Anchor the cube to the palm or keep it centered.",
    )
    return parser.parse_args()


def _backend_from_name(name: str) -> int | None:
    if name == "auto":
        if sys.platform == "darwin":
            return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
        return cv2.CAP_ANY
    if name == "any":
        return cv2.CAP_ANY
    if name == "avfoundation":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    return cv2.CAP_ANY


def _open_camera(index: int, backend: int | None) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(current: float, target: float, alpha: float) -> float:
    return current + alpha * (target - current)


def _is_black_frame(frame) -> bool:
    if frame is None or frame.size == 0:
        return True
    mean = float(frame.mean())
    std = float(frame.std())
    return mean < 2.0 and std < 2.0


def _black_frame_message(index: int, backend: str) -> str:
    return (
        "Camera frames are black. On macOS this usually means camera permission is "
        "blocked for Terminal/Python or the wrong camera index is selected. "
        "Check System Settings > Privacy & Security > Camera, then try "
        f"`--camera-index {index + 1}` or `--auto-camera`. "
        f"Current backend: {backend}."
    )


def _rotate_point(point: Tuple[float, float, float], yaw: float, pitch: float) -> Tuple[float, float, float]:
    x, y, z = point
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    xz = x * cos_y + z * sin_y
    zz = -x * sin_y + z * cos_y
    yz = y * cos_p - zz * sin_p
    zz = y * sin_p + zz * cos_p
    return xz, yz, zz


def _project_cube(
    center: Tuple[int, int],
    size: float,
    yaw_deg: float,
    pitch_deg: float,
    focal_length: float,
    distance: float,
) -> Tuple[list[Tuple[int, int]], list[Tuple[float, float, float]]]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cx, cy = center
    projected: list[Tuple[int, int]] = []
    rotated: list[Tuple[float, float, float]] = []
    for vx, vy, vz in BASE_VERTICES:
        point = (vx * size, vy * size, vz * size)
        rx, ry, rz = _rotate_point(point, yaw, pitch)
        rz += distance
        rotated.append((rx, ry, rz))
        if rz <= 0.01:
            rz = 0.01
        u = int(cx + (focal_length * rx / rz))
        v = int(cy + (focal_length * ry / rz))
        projected.append((u, v))
    return projected, rotated


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


def _draw_cube(
    frame,
    projected: list[Tuple[int, int]],
    rotated: list[Tuple[float, float, float]],
    highlight: bool,
) -> None:
    light_dir = _normalize((0.6, 0.9, 0.4))
    face_depths = []
    for face_indices, base_color in FACES:
        verts = [rotated[idx] for idx in face_indices]
        avg_z = sum(v[2] for v in verts) / len(verts)
        normal = _face_normal(verts[0], verts[1], verts[2])
        intensity = max(
            normal[0] * light_dir[0] + normal[1] * light_dir[1] + normal[2] * light_dir[2],
            0.2,
        )
        shade = tuple(int(c * intensity) for c in base_color)
        face_depths.append((avg_z, face_indices, shade))

    for _avg_z, face_indices, shade in sorted(face_depths, key=lambda item: item[0], reverse=True):
        pts = [projected[idx] for idx in face_indices]
        cv2.fillConvexPoly(frame, _to_poly(pts), shade)

    if highlight:
        edge_color = (0, 220, 255)
        for start, end in EDGES:
            cv2.line(frame, projected[start], projected[end], edge_color, 2)


def _cube_bounds(projected: list[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    return min(xs), min(ys), max(xs), max(ys)


def _to_poly(points: list[Tuple[int, int]]):
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def _draw_trail(frame, trail: Deque[Tuple[int, int]]) -> None:
    if len(trail) < 2:
        return
    points = list(trail)
    for i in range(1, len(points)):
        alpha = i / len(points)
        color = (int(255 * (1 - alpha)), int(120 + 100 * alpha), 255)
        cv2.line(frame, points[i - 1], points[i], color, 2)


def main() -> None:
    args = parse_args()
    state = CubeState(trail=deque(maxlen=args.trail_length))

    backend = _backend_from_name(args.backend)
    cap = _open_camera(args.camera_index, backend)
    camera_index = args.camera_index
    if not cap.isOpened():
        if args.auto_camera:
            for idx in range(args.camera_index + 1, args.max_camera_index + 1):
                cap = _open_camera(idx, backend)
                if cap.isOpened():
                    camera_index = idx
                    break
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera. Try a different --camera-index.")

    tracker = HandTracker(
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_path=args.model_path,
    )

    try:
        black_frames = 0
        max_black_frames = 60
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            if state.center is None:
                state.center = (width // 2, height // 2)
            focal_length = width * 0.9

            if _is_black_frame(frame):
                black_frames += 1
            else:
                black_frames = 0

            if black_frames >= max_black_frames:
                message = _black_frame_message(camera_index, args.backend)
                print(message)
                if args.auto_camera and camera_index < args.max_camera_index:
                    cap.release()
                    camera_index += 1
                    cap = _open_camera(camera_index, backend)
                    black_frames = 0
                    continue
                raise RuntimeError(message)

            results = tracker.process(frame)
            hand_results = None
            tip = None
            pinch = False
            palm_center = None
            if results:
                hand_results = results
                hand = results[0]
                tip = hand.landmarks_2d[8]
                thumb_tip = hand.landmarks_2d[4]
                dx = tip[0] - thumb_tip[0]
                dy = tip[1] - thumb_tip[1]
                pinch_distance = math.hypot(dx, dy)
                pinch_threshold = max(30.0, width * 0.05)
                pinch = pinch_distance < pinch_threshold
                state.trail.append(tip)
                state.last_hand_time = time.time()

                if args.cube_anchor == "hand":
                    palm_indices = (0, 5, 9, 13, 17)
                    avg_x = (
                        sum(hand.landmarks_2d[i][0] for i in palm_indices)
                        / len(palm_indices)
                    )
                    avg_y = (
                        sum(hand.landmarks_2d[i][1] for i in palm_indices)
                        / len(palm_indices)
                    )
                    palm_center = (int(avg_x), int(avg_y))

            else:
                if time.time() - state.last_hand_time > 0.5:
                    state.trail.clear()
                    state.dragging = False
                    state.last_tip = None

            projected, rotated = _project_cube(
                state.center or (width // 2, height // 2),
                args.cube_size,
                state.yaw,
                state.pitch,
                focal_length,
                args.cube_distance,
            )
            bounds = _cube_bounds(projected)

            inside = False
            if tip:
                min_x, min_y, max_x, max_y = bounds
                padding = 12
                inside = (
                    (min_x - padding)
                    <= tip[0]
                    <= (max_x + padding)
                    and (min_y - padding)
                    <= tip[1]
                    <= (max_y + padding)
                )
                if pinch and inside:
                    if state.last_tip is not None:
                        dx = tip[0] - state.last_tip[0]
                        dy = tip[1] - state.last_tip[1]
                        state.yaw_target += dx * args.sensitivity
                        state.pitch_target -= dy * args.sensitivity
                        state.pitch_target = _clamp(state.pitch_target, -80.0, 80.0)
                    state.dragging = True
                else:
                    state.dragging = False
                state.last_tip = tip

            if args.cube_anchor == "center":
                state.center = (width // 2, height // 2)
            elif palm_center is not None:
                target_center = palm_center
                anchor_smoothing = args.anchor_smoothing
                if pinch and inside and tip:
                    blend = 0.65
                    target_center = (
                        int(palm_center[0] * (1 - blend) + tip[0] * blend),
                        int(palm_center[1] * (1 - blend) + tip[1] * blend),
                    )
                    anchor_smoothing = max(anchor_smoothing, 0.22)
                if state.center is None:
                    state.center = target_center
                else:
                    delta_x = target_center[0] - state.center[0]
                    delta_y = target_center[1] - state.center[1]
                    distance = math.hypot(delta_x, delta_y)
                    if distance > args.max_anchor_step and distance > 0:
                        scale = args.max_anchor_step / distance
                        target_center = (
                            int(state.center[0] + delta_x * scale),
                            int(state.center[1] + delta_y * scale),
                        )
                    state.center = (
                        int(_lerp(state.center[0], target_center[0], anchor_smoothing)),
                        int(_lerp(state.center[1], target_center[1], anchor_smoothing)),
                    )

            active_rotation_smoothing = args.rotation_smoothing
            if state.dragging:
                active_rotation_smoothing = max(active_rotation_smoothing, 0.25)
            state.yaw = _lerp(state.yaw, state.yaw_target, active_rotation_smoothing)
            state.pitch = _lerp(state.pitch, state.pitch_target, active_rotation_smoothing)

            projected, rotated = _project_cube(
                state.center or (width // 2, height // 2),
                args.cube_size,
                state.yaw,
                state.pitch,
                focal_length,
                args.cube_distance,
            )

            _draw_cube(frame, projected, rotated, state.dragging)

            if hand_results:
                tracker.draw(frame, hand_results)

            if tip:
                cv2.circle(frame, tip, 6, (0, 255, 255) if pinch else (0, 180, 255), -1)

            _draw_trail(frame, state.trail)

            if pinch and tip:
                cv2.putText(
                    frame,
                    "touch + drag to rotate",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Hand Cube", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
