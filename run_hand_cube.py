#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cv3d.cube_render import CubeRenderer
from cv3d.data_logger import DataLogger, LoggerConfig
from cv3d.gesture_model import GestureConfig, GestureTracker
from cv3d.hand_input import HandInput
from cv3d.hand_menu import HandMenu, MenuItem
from cv3d.orb_render import OrbRenderer
from cv3d.palette import (
    IOS_BLUE,
    IOS_BLUE_SOFT,
    IOS_BORDER,
    IOS_GLASS,
    IOS_TEXT,
)
from cv3d.pipeline import HandWorker, ThreadedCapture
from cv3d.physics import CubePhysics, CubeState, PhysicsConfig, compute_dt
from cv3d.ui import draw_glass_panel, draw_rounded_rect, draw_text


def _parse_hsv(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV values must be in H,S,V format.")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV values must be integers.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay 3D cubes on the camera feed and push them with your hand."
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
    parser.add_argument(
        "--input-mode",
        choices=("mediapipe", "blue-glove", "hybrid"),
        default="hybrid",
        help="Hand input mode; use blue-glove for colored glove tracking.",
    )
    parser.add_argument(
        "--glove-lower",
        type=_parse_hsv,
        default=(90, 60, 50),
        help="Lower HSV bound for blue glove detection (H,S,V).",
    )
    parser.add_argument(
        "--glove-upper",
        type=_parse_hsv,
        default=(130, 255, 255),
        help="Upper HSV bound for blue glove detection (H,S,V).",
    )
    parser.add_argument(
        "--glove-min-area",
        type=float,
        default=800.0,
        help="Minimum contour area for glove detection.",
    )
    parser.add_argument(
        "--glove-kernel",
        type=int,
        default=3,
        help="Kernel size for glove mask morphology.",
    )
    parser.add_argument(
        "--glove-open",
        type=int,
        default=0,
        help="Morphology open iterations for glove mask.",
    )
    parser.add_argument(
        "--glove-close",
        type=int,
        default=1,
        help="Morphology close iterations for glove mask.",
    )
    parser.add_argument(
        "--glove-dilate",
        type=int,
        default=1,
        help="Dilation iterations for glove mask.",
    )
    parser.add_argument(
        "--glove-contour-epsilon",
        type=float,
        default=0.008,
        help="Contour approximation epsilon (fraction of arc length).",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Show the glove segmentation mask for tuning HSV bounds.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to write a logging session (JSONL + optional images).",
    )
    parser.add_argument(
        "--log-frames",
        action="store_true",
        help="Save raw camera frames as JPEGs.",
    )
    parser.add_argument(
        "--log-overlay",
        action="store_true",
        help="Save overlay frames (with UI/cubes) as JPEGs.",
    )
    parser.add_argument(
        "--log-mask",
        action="store_true",
        help="Save hand masks as PNGs when available.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log every Nth frame.",
    )
    parser.add_argument(
        "--log-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for logged frames (0-100).",
    )
    parser.add_argument(
        "--log-queue",
        type=int,
        default=256,
        help="Max queued log items before dropping frames.",
    )
    parser.add_argument(
        "--show-gestures",
        action="store_true",
        help="Overlay the gesture model labels near each hand.",
    )
    parser.add_argument(
        "--gesture-smoothing",
        type=float,
        default=0.6,
        help="Smoothing factor for gesture classification (0-1).",
    )
    parser.add_argument(
        "--gesture-min-confidence",
        type=float,
        default=0.25,
        help="Minimum confidence required to show a gesture label.",
    )
    parser.add_argument(
        "--no-threaded",
        action="store_true",
        help="Disable threaded camera capture.",
    )
    parser.add_argument(
        "--no-hand-worker",
        action="store_true",
        help="Process hand tracking on the main thread.",
    )
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--pinch-ratio",
        type=float,
        default=0.45,
        help="Pinch threshold as a fraction of palm size.",
    )
    parser.add_argument(
        "--open-ratio",
        type=float,
        default=1.6,
        help="Open palm threshold as a ratio of fingertip distance to palm size.",
    )
    parser.add_argument(
        "--press-depth",
        type=float,
        default=0.05,
        help="Depth threshold for a fingertip press gesture.",
    )
    parser.add_argument(
        "--menu-scale",
        type=float,
        default=2.5,
        help="Scale the size of the hand menu.",
    )
    parser.add_argument(
        "--menu-open-hold",
        type=float,
        default=1.6,
        help="Seconds to hold the open-palm gesture to toggle the menu.",
    )
    parser.add_argument(
        "--menu-open-strength",
        type=float,
        default=0.35,
        help="Minimum open-palm strength to allow menu toggle.",
    )
    parser.add_argument(
        "--grip-ratio",
        type=float,
        default=1.1,
        help="Grip threshold as a ratio of fingertip distance to palm size.",
    )
    parser.add_argument(
        "--anchor-smoothing",
        type=float,
        default=None,
        help="Deprecated: no longer used.",
    )
    parser.add_argument(
        "--max-anchor-step",
        type=float,
        default=None,
        help="Deprecated: no longer used.",
    )
    parser.add_argument(
        "--cube-anchor",
        choices=("hand", "center"),
        default=None,
        help="Deprecated: no longer used.",
    )
    parser.add_argument("--contact-force", type=float, default=900.0)
    parser.add_argument("--hand-velocity-scale", type=float, default=0.7)
    parser.add_argument("--contact-distance", type=float, default=8.0)
    parser.add_argument("--gravity", type=float, default=1600.0)
    parser.add_argument("--grab-distance", type=float, default=120.0)
    parser.add_argument("--grab-strength", type=float, default=30.0)
    parser.add_argument("--grab-damping", type=float, default=0.85)
    parser.add_argument("--grab-follow", type=float, default=0.7)
    parser.add_argument("--damping", type=float, default=0.92)
    parser.add_argument("--restitution", type=float, default=0.86)
    parser.add_argument("--max-speed", type=float, default=1400.0)
    parser.add_argument("--rotation-smoothing", type=float, default=0.2)
    parser.add_argument("--spin-strength", type=float, default=0.2)
    parser.add_argument("--num-cubes", type=int, default=3)
    parser.add_argument("--cube-size", type=float, default=0.7)
    parser.add_argument("--cube-distance", type=float, default=5.8)
    parser.add_argument("--num-orbs", type=int, default=1)
    parser.add_argument("--orb-size", type=float, default=0.5)
    parser.add_argument("--max-orbs", type=int, default=6)
    parser.add_argument(
        "--power-shockwave",
        type=float,
        default=1200.0,
        help="Shockwave impulse strength for fist gesture (0 disables).",
    )
    parser.add_argument(
        "--power-shockwave-radius",
        type=float,
        default=260.0,
        help="Shockwave radius in pixels.",
    )
    parser.add_argument(
        "--power-shockwave-cooldown",
        type=float,
        default=0.6,
        help="Cooldown between fist shockwaves.",
    )
    parser.add_argument(
        "--power-tractor",
        type=float,
        default=520.0,
        help="Tractor pull strength for open-palm gesture (0 disables).",
    )
    parser.add_argument(
        "--power-tractor-radius",
        type=float,
        default=280.0,
        help="Tractor pull radius in pixels.",
    )
    parser.add_argument(
        "--power-laser",
        type=float,
        default=900.0,
        help="Laser push strength for point gesture (0 disables).",
    )
    parser.add_argument(
        "--power-laser-range",
        type=float,
        default=420.0,
        help="Laser range in pixels.",
    )
    parser.add_argument(
        "--power-laser-width",
        type=float,
        default=60.0,
        help="Laser width in pixels.",
    )
    parser.add_argument(
        "--power-spawn-cooldown",
        type=float,
        default=0.9,
        help="Cooldown between two-finger spawns.",
    )
    parser.add_argument(
        "--power-spawn-velocity",
        type=float,
        default=0.8,
        help="Velocity scale for spawned orbs (relative to hand velocity).",
    )
    return parser.parse_args()


@dataclass
class SceneObject:
    kind: str
    state: CubeState
    size: float


@dataclass
class PowerConfig:
    shockwave_strength: float
    shockwave_radius: float
    shockwave_cooldown: float
    tractor_strength: float
    tractor_radius: float
    laser_strength: float
    laser_range: float
    laser_width: float
    spawn_cooldown: float
    spawn_velocity_scale: float
    max_orbs: int
    orb_size: float


def _make_object(kind: str, size: float) -> SceneObject:
    return SceneObject(kind=kind, state=CubeState(), size=size)


def _count_objects(objects: list[SceneObject], kind: str) -> int:
    return sum(1 for obj in objects if obj.kind == kind)


def _set_object_count(
    objects: list[SceneObject], kind: str, target: int, size: float
) -> None:
    target = max(0, int(target))
    current = _count_objects(objects, kind)
    if target > current:
        for _ in range(target - current):
            objects.append(_make_object(kind, size))
    elif target < current:
        remove = current - target
        for idx in range(len(objects) - 1, -1, -1):
            if objects[idx].kind != kind:
                continue
            del objects[idx]
            remove -= 1
            if remove <= 0:
                break


def _avg_scale(scale: tuple[float, float, float]) -> float:
    return max(0.2, (scale[0] + scale[1] + scale[2]) / 3.0)


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


def _set_cube_count(objects: list[SceneObject], value: float, cube_size: float) -> None:
    target = int(round(value))
    target = max(1, min(target, 5))
    _set_object_count(objects, "cube", target, cube_size)


def _set_orb_count(
    objects: list[SceneObject], value: float, orb_size: float, max_orbs: int
) -> None:
    target = int(round(value))
    target = max(0, min(target, max_orbs))
    _set_object_count(objects, "orb", target, orb_size)


def _draw_stats(frame, lines, anchor=(18, 18)) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 10
    line_height = 20
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    width = max(size[0] for size in text_sizes) + padding * 2
    height = line_height * len(lines) + padding * 2
    x, y = anchor
    panel = (x, y, width, height)
    radius = max(10, int(min(width, height) * 0.08))
    draw_glass_panel(
        frame,
        panel,
        radius,
        IOS_GLASS,
        IOS_BORDER,
        border_thickness=1,
        tint_alpha=0.28,
        blur_sigma=12.0,
        shadow=True,
        shadow_alpha=0.2,
    )
    for idx, line in enumerate(lines):
        text_y = y + padding + line_height * (idx + 1) - 4
        draw_text(
            frame,
            line,
            (x + padding, text_y),
            font,
            font_scale,
            IOS_TEXT,
            1,
            shadow=False,
        )


def _draw_graph(frame, values, label: str, anchor=(18, 220)) -> None:
    if len(values) < 2:
        return
    panel_w = 260
    panel_h = 140
    padding = 12
    x, y = anchor
    panel = (x, y, panel_w, panel_h)
    radius = max(10, int(min(panel_w, panel_h) * 0.08))
    draw_glass_panel(
        frame,
        panel,
        radius,
        IOS_GLASS,
        IOS_BORDER,
        border_thickness=1,
        tint_alpha=0.28,
        blur_sigma=12.0,
        shadow=True,
        shadow_alpha=0.2,
    )
    draw_text(
        frame,
        label,
        (x + padding, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        IOS_TEXT,
        1,
        shadow=False,
    )
    graph_left = x + padding
    graph_right = x + panel_w - padding
    graph_top = y + 30
    graph_bottom = y + panel_h - padding
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    step = (graph_right - graph_left) / float(len(values) - 1)
    points = []
    for idx, value in enumerate(values):
        ratio = max(0.0, min(1.0, value / max_value))
        px = int(graph_left + idx * step)
        py = int(graph_bottom - ratio * (graph_bottom - graph_top))
        points.append((px, py))
    if len(points) >= 2:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, IOS_BLUE, 2)


def _draw_gesture_labels(frame, hands, results, min_confidence: float) -> None:
    if not results:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    padding = 6
    frame_h, frame_w = frame.shape[:2]
    for hand in hands:
        result = results.get(hand.id)
        if result is None or result.label == "unknown":
            continue
        if result.confidence < min_confidence:
            continue
        label = result.label.upper()
        text_size = cv2.getTextSize(label, font, scale, 1)[0]
        box_w = text_size[0] + padding * 2
        box_h = text_size[1] + padding * 2
        pointer = hand.center
        if getattr(hand, "landmarks_2d", None) and len(hand.landmarks_2d) > 8:
            pointer = hand.landmarks_2d[8]
        x = int(pointer[0] - box_w / 2)
        y = int(pointer[1] - box_h - 16)
        x = max(8, min(x, frame_w - box_w - 8))
        y = max(8, min(y, frame_h - box_h - 8))
        rect = (x, y, box_w, box_h)
        draw_glass_panel(
            frame,
            rect,
            max(6, int(box_h * 0.4)),
            IOS_GLASS,
            IOS_BORDER,
            border_thickness=1,
            tint_alpha=0.28,
            blur_sigma=10.0,
            shadow=True,
            shadow_alpha=0.2,
        )
        draw_text(
            frame,
            label,
            (x + padding, y + padding + text_size[1]),
            font,
            scale,
            IOS_TEXT,
            1,
            shadow=False,
        )


PALM_INDICES = (0, 5, 9, 13, 17)


def _hand_palm_center(hand) -> tuple[float, float]:
    points = getattr(hand, "landmarks_2d", None)
    if points and len(points) >= 21:
        return (
            sum(points[idx][0] for idx in PALM_INDICES) / len(PALM_INDICES),
            sum(points[idx][1] for idx in PALM_INDICES) / len(PALM_INDICES),
        )
    return hand.center


def _apply_shockwave(
    objects: list[SceneObject],
    origin: tuple[float, float],
    strength: float,
    radius: float,
    dt: float,
) -> None:
    if strength <= 0.0 or radius <= 0.0:
        return
    for obj in objects:
        state = obj.state
        if state.position is None:
            continue
        dx = state.position[0] - origin[0]
        dy = state.position[1] - origin[1]
        dist = math.hypot(dx, dy)
        if dist <= 1e-3 or dist > radius:
            continue
        nx, ny = dx / dist, dy / dist
        scale = 1.0 - dist / radius
        impulse = strength * scale * dt
        state.velocity = (state.velocity[0] + nx * impulse, state.velocity[1] + ny * impulse)
        state.yaw_target += nx * impulse * 0.01
        state.pitch_target -= ny * impulse * 0.01


def _apply_tractor(
    objects: list[SceneObject],
    origin: tuple[float, float],
    strength: float,
    radius: float,
    dt: float,
) -> None:
    if strength <= 0.0 or radius <= 0.0:
        return
    for obj in objects:
        state = obj.state
        if state.position is None:
            continue
        dx = origin[0] - state.position[0]
        dy = origin[1] - state.position[1]
        dist = math.hypot(dx, dy)
        if dist <= 1e-3 or dist > radius:
            continue
        nx, ny = dx / dist, dy / dist
        scale = 1.0 - dist / radius
        impulse = strength * scale * dt
        state.velocity = (state.velocity[0] + nx * impulse, state.velocity[1] + ny * impulse)


def _apply_laser(
    objects: list[SceneObject],
    origin: tuple[float, float],
    direction: tuple[float, float],
    strength: float,
    laser_range: float,
    width: float,
    dt: float,
) -> None:
    if strength <= 0.0 or laser_range <= 0.0 or width <= 0.0:
        return
    dir_x, dir_y = direction
    for obj in objects:
        state = obj.state
        if state.position is None:
            continue
        to_x = state.position[0] - origin[0]
        to_y = state.position[1] - origin[1]
        proj = to_x * dir_x + to_y * dir_y
        if proj <= 0.0 or proj > laser_range:
            continue
        dist_sq = to_x * to_x + to_y * to_y
        perp_sq = max(0.0, dist_sq - proj * proj)
        perp = math.sqrt(perp_sq)
        if perp > width:
            continue
        scale = (1.0 - proj / laser_range) * (1.0 - perp / width)
        impulse = strength * scale * dt
        state.velocity = (
            state.velocity[0] + dir_x * impulse,
            state.velocity[1] + dir_y * impulse,
        )


def _spawn_orb(
    objects: list[SceneObject],
    origin: tuple[float, float],
    velocity: tuple[float, float],
    size: float,
    max_orbs: int,
) -> bool:
    if _count_objects(objects, "orb") >= max_orbs:
        return False
    obj = _make_object("orb", size)
    obj.state.position = (origin[0], origin[1])
    obj.state.velocity = velocity
    obj.state.yaw = 0.0
    obj.state.pitch = 0.0
    obj.state.yaw_target = 0.0
    obj.state.pitch_target = 0.0
    objects.append(obj)
    return True


def _apply_gesture_powers(
    objects: list[SceneObject],
    hands,
    gestures,
    gesture_state: dict[int, dict[str, float | str]],
    now: float,
    dt: float,
    config: PowerConfig,
    min_confidence: float,
) -> None:
    if not gestures:
        return
    active_ids = set()
    spawn_requests = 0
    for hand in hands:
        result = gestures.get(hand.id)
        if result is None or result.label == "unknown":
            continue
        if result.confidence < min_confidence:
            continue
        active_ids.add(hand.id)
        state = gesture_state.setdefault(
            hand.id, {"label": "unknown", "last_shockwave": 0.0, "last_spawn": 0.0}
        )
        prev_label = state["label"]
        label = result.label
        state["label"] = label

        if (
            label == "fist"
            and prev_label != "fist"
            and now - float(state["last_shockwave"]) > config.shockwave_cooldown
        ):
            _apply_shockwave(
                objects,
                hand.center,
                config.shockwave_strength,
                config.shockwave_radius,
                dt,
            )
            state["last_shockwave"] = now

        if (
            label == "two"
            and prev_label != "two"
            and now - float(state["last_spawn"]) > config.spawn_cooldown
        ):
            spawn_requests += 1
            state["last_spawn"] = now

        if label == "open":
            _apply_tractor(
                objects,
                hand.center,
                config.tractor_strength,
                config.tractor_radius,
                dt,
            )

        if label == "point":
            points = getattr(hand, "landmarks_2d", None)
            if points and len(points) > 8:
                palm = _hand_palm_center(hand)
                tip = points[8]
                dir_x = tip[0] - palm[0]
                dir_y = tip[1] - palm[1]
                length = math.hypot(dir_x, dir_y)
                if length > 1.0:
                    dir_x /= length
                    dir_y /= length
                    _apply_laser(
                        objects,
                        (tip[0], tip[1]),
                        (dir_x, dir_y),
                        config.laser_strength,
                        config.laser_range,
                        config.laser_width,
                        dt,
                    )

    if spawn_requests > 0:
        for _ in range(spawn_requests):
            if _count_objects(objects, "orb") >= config.max_orbs:
                break
            for hand in hands:
                result = gestures.get(hand.id)
                if result is None or result.label != "two":
                    continue
                velocity = (
                    hand.velocity[0] * config.spawn_velocity_scale,
                    hand.velocity[1] * config.spawn_velocity_scale,
                )
                _spawn_orb(objects, hand.center, velocity, config.orb_size, config.max_orbs)
                break

    stale_ids = [hand_id for hand_id in gesture_state if hand_id not in active_ids]
    for hand_id in stale_ids:
        gesture_state.pop(hand_id, None)


def _resolve_cube_collisions(states: list[CubeState], radii: list[float], restitution: float) -> None:
    count = len(states)
    if count < 2:
        return
    for i in range(count):
        state_a = states[i]
        if state_a.position is None:
            continue
        for j in range(i + 1, count):
            state_b = states[j]
            if state_b.position is None:
                continue
            ax, ay = state_a.position
            bx, by = state_b.position
            dx = bx - ax
            dy = by - ay
            dist = math.hypot(dx, dy)
            min_dist = radii[i] + radii[j]
            if dist < 1e-6:
                nx, ny = 1.0, 0.0
                dist = 1.0
            else:
                nx, ny = dx / dist, dy / dist
            if dist >= min_dist:
                continue

            overlap = min_dist - dist
            ax -= nx * overlap * 0.5
            ay -= ny * overlap * 0.5
            bx += nx * overlap * 0.5
            by += ny * overlap * 0.5
            state_a.position = (ax, ay)
            state_b.position = (bx, by)

            avx, avy = state_a.velocity
            bvx, bvy = state_b.velocity
            rvx = bvx - avx
            rvy = bvy - avy
            vel_along = rvx * nx + rvy * ny
            if vel_along < 0:
                impulse = -(1.0 + restitution) * vel_along / 2.0
                avx -= impulse * nx
                avy -= impulse * ny
                bvx += impulse * nx
                bvy += impulse * ny
                state_a.velocity = (avx, avy)
                state_b.velocity = (bvx, bvy)

            tangential = rvx * -ny + rvy * nx
            spin = tangential * 0.02
            state_a.yaw_target += spin
            state_b.yaw_target -= spin


def _point_in_rect(point: tuple[float, float], rect: tuple[int, int, int, int]) -> bool:
    x, y, w, h = rect
    return x <= point[0] <= x + w and y <= point[1] <= y + h


def _top_bar_layout(frame_shape) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    button_w = int(max(96, width * 0.13))
    button_h = int(max(34, height * 0.05))
    x = width - button_w - int(max(16, width * 0.03))
    y = int(max(14, height * 0.025))
    return (x, y, button_w, button_h)


def _draw_top_bar(frame, menu_open: bool, hover: bool) -> tuple[int, int, int, int]:
    button_rect = _top_bar_layout(frame.shape)
    x, y, w, h = button_rect
    radius = max(1, h // 2)
    draw_glass_panel(
        frame,
        button_rect,
        radius,
        IOS_GLASS,
        IOS_BORDER,
        border_thickness=1,
        tint_alpha=0.3,
        blur_sigma=12.0,
        shadow=True,
        shadow_alpha=0.22,
    )
    if menu_open or hover:
        accent = IOS_BLUE if menu_open else IOS_BLUE_SOFT
        draw_rounded_rect(frame, button_rect, radius, accent, 2)
    label = "Controls" if not menu_open else "Close"
    font_scale = 0.55
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    draw_text(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        IOS_TEXT,
        2,
        shadow=False,
    )
    return button_rect


def _handle_menu_button(
    hands, frame_shape, last_toggle: float, cooldown: float, menu: HandMenu, now: float
) -> tuple[bool, float]:
    button_rect = _top_bar_layout(frame_shape)
    hover = False
    for hand in hands:
        pointer = hand.center
        if getattr(hand, "landmarks_2d", None) and len(hand.landmarks_2d) > 8:
            pointer = hand.landmarks_2d[8]
        if _point_in_rect(pointer, button_rect):
            hover = True
            if getattr(hand, "press", False) and now - last_toggle > cooldown:
                menu.toggle(now)
                return True, now
    return hover, last_toggle


def main() -> None:
    args = parse_args()

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

    threaded = not args.no_threaded

    hand_input = HandInput(
        max_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_path=args.model_path,
        mode=args.input_mode,
        glove_lower=args.glove_lower,
        glove_upper=args.glove_upper,
        glove_min_area=args.glove_min_area,
        glove_kernel_size=args.glove_kernel,
        glove_open=args.glove_open,
        glove_close=args.glove_close,
        glove_dilate=args.glove_dilate,
        glove_contour_epsilon=args.glove_contour_epsilon,
        pinch_ratio=args.pinch_ratio,
        grip_ratio=args.grip_ratio,
        open_ratio=args.open_ratio,
        press_depth=args.press_depth,
    )
    renderer = CubeRenderer(size=args.cube_size, distance=args.cube_distance)
    orb_renderer = OrbRenderer(distance=args.cube_distance)
    physics = CubePhysics(
        PhysicsConfig(
            contact_force=args.contact_force,
            hand_velocity_scale=args.hand_velocity_scale,
            contact_distance=args.contact_distance,
            gravity=args.gravity,
            grab_distance=args.grab_distance,
            grab_strength=args.grab_strength,
            grab_damping=args.grab_damping,
            grab_follow=args.grab_follow,
            depth_scale_strength=0.0,
            damping=args.damping,
            restitution=args.restitution,
            max_speed=args.max_speed,
            rotation_smoothing=args.rotation_smoothing,
            spin_strength=args.spin_strength,
        )
    )
    gesture_tracker = GestureTracker(
        GestureConfig(
            smoothing=args.gesture_smoothing,
            min_confidence=args.gesture_min_confidence,
        )
    )
    capture = ThreadedCapture(cap).start() if threaded else None
    hand_worker = HandWorker(hand_input).start() if threaded and not args.no_hand_worker else None
    logger = None
    if args.log_dir is not None:
        logger = DataLogger(
            LoggerConfig(
                root_dir=args.log_dir,
                record_frames=args.log_frames,
                record_overlay=args.log_overlay,
                record_mask=args.log_mask,
                every=args.log_every,
                jpeg_quality=args.log_jpeg_quality,
                queue_size=args.log_queue,
            ),
            metadata={"args": vars(args)},
        ).start()
    objects: list[SceneObject] = []
    _set_object_count(objects, "cube", max(1, args.num_cubes), args.cube_size)
    _set_object_count(
        objects,
        "orb",
        max(0, min(args.num_orbs, args.max_orbs)),
        args.orb_size,
    )
    clock_state = CubeState()
    ui_flags = {"stats": False, "graphs": False}
    speed_history = deque(maxlen=120)
    menu_toggle_time = 0.0
    menu_hover = False
    fps_value = 0.0
    gesture_state: dict[int, dict[str, float | str]] = {}
    power_config = PowerConfig(
        shockwave_strength=args.power_shockwave,
        shockwave_radius=args.power_shockwave_radius,
        shockwave_cooldown=args.power_shockwave_cooldown,
        tractor_strength=args.power_tractor,
        tractor_radius=args.power_tractor_radius,
        laser_strength=args.power_laser,
        laser_range=args.power_laser_range,
        laser_width=args.power_laser_width,
        spawn_cooldown=args.power_spawn_cooldown,
        spawn_velocity_scale=args.power_spawn_velocity,
        max_orbs=max(0, args.max_orbs),
        orb_size=args.orb_size,
    )

    def _toggle_stats() -> None:
        ui_flags["stats"] = not ui_flags["stats"]

    def _toggle_graphs() -> None:
        ui_flags["graphs"] = not ui_flags["graphs"]

    menu = HandMenu(
        [
            MenuItem(
                "Gravity",
                lambda: physics.config.gravity,
                lambda value: setattr(physics.config, "gravity", max(0.0, float(value))),
                0.0,
                2000.0,
                50.0,
                "{:.0f}",
            ),
            MenuItem(
                "Contact",
                lambda: physics.config.contact_force,
                lambda value: setattr(physics.config, "contact_force", max(0.0, float(value))),
                300.0,
                2000.0,
                50.0,
                "{:.0f}",
            ),
            MenuItem(
                "Cubes",
                lambda: float(_count_objects(objects, "cube")),
                lambda value: _set_cube_count(objects, value, args.cube_size),
                1.0,
                5.0,
                1.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "Orbs",
                lambda: float(_count_objects(objects, "orb")),
                lambda value: _set_orb_count(objects, value, args.orb_size, args.max_orbs),
                0.0,
                float(max(0, args.max_orbs)),
                1.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "Pinch",
                hand_input.get_pinch_ratio,
                hand_input.set_pinch_ratio,
                0.35,
                0.6,
                0.02,
                "{:.2f}",
            ),
            MenuItem(
                "Grip",
                hand_input.get_grip_ratio,
                hand_input.set_grip_ratio,
                0.8,
                1.4,
                0.05,
                "{:.2f}",
            ),
            MenuItem(
                "Stats",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_stats,
                state=lambda: ui_flags["stats"],
            ),
            MenuItem(
                "Graphs",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_graphs,
                state=lambda: ui_flags["graphs"],
            ),
        ],
        open_hold=args.menu_open_hold,
        toggle_cooldown=1.0,
        open_strength_threshold=args.menu_open_strength,
        scale=args.menu_scale,
        use_open_gesture=False,
    )

    def _restart_capture(index: int) -> bool:
        nonlocal cap, capture
        if capture is not None:
            capture.release()
            capture = None
        else:
            cap.release()
        cap = _open_camera(index, backend)
        if not cap.isOpened():
            return False
        if threaded:
            capture = ThreadedCapture(cap).start()
        return True

    try:
        black_frames = 0
        max_black_frames = 60
        frame_id_counter = 0
        while True:
            if capture is not None:
                ok, frame, frame_id = capture.read()
            else:
                ok, frame = cap.read()
                if ok:
                    frame_id_counter += 1
                frame_id = frame_id_counter
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            base_frame = frame.copy()

            height, width = frame.shape[:2]
            focal_length = width * 0.9

            if _is_black_frame(frame):
                black_frames += 1
            else:
                black_frames = 0

            if black_frames >= max_black_frames:
                message = _black_frame_message(camera_index, args.backend)
                print(message)
                if args.auto_camera and camera_index < args.max_camera_index:
                    camera_index += 1
                    if not _restart_capture(camera_index):
                        raise RuntimeError(message)
                    black_frames = 0
                    continue
                raise RuntimeError(message)

            now = time.time()
            dt = compute_dt(clock_state, now)
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_value = inst_fps if fps_value == 0.0 else fps_value * 0.9 + inst_fps * 0.1

            if hand_worker is not None:
                hand_worker.submit(base_frame, frame_id)
                hands, _ = hand_worker.get()
            else:
                hands = hand_input.update(base_frame)
            gesture_results = gesture_tracker.update(hands, now)
            menu.update(hands, now, frame.shape)
            menu_hover, menu_toggle_time = _handle_menu_button(
                hands, frame.shape, menu_toggle_time, 0.35, menu, now
            )
            menu.handle_input(hands, frame.shape, now)

            current_count = max(1, len(objects))
            contact_any = False
            for idx, obj in enumerate(objects):
                state = obj.state
                if state.position is None:
                    if current_count == 1:
                        start_x = width / 2.0
                    else:
                        spacing = width / float(current_count + 1)
                        start_x = spacing * (idx + 1)
                    start_y = height * (0.35 + 0.05 * (idx % 2))
                    state.position = (start_x, start_y)
                    state.yaw = 20.0 + idx * 12.0
                    state.pitch = -15.0 + idx * 6.0
                    state.yaw_target = state.yaw
                    state.pitch_target = state.pitch
                    state.velocity = (0.0, 0.0)

            _apply_gesture_powers(
                objects,
                hands,
                gesture_results,
                gesture_state,
                now,
                dt,
                power_config,
                max(args.gesture_min_confidence, 0.35),
            )

            contact_flags: list[bool] = []
            radii: list[float] = []
            object_states: list[CubeState] = []
            for obj in objects:
                state = obj.state
                if state.position is None:
                    contact_flags.append(False)
                    radii.append(0.0)
                    object_states.append(state)
                    continue

                if obj.kind == "cube":
                    projected, _ = renderer.project(
                        (int(state.position[0]), int(state.position[1])),
                        state.yaw,
                        state.pitch,
                        focal_length,
                        state.scale,
                    )
                    min_x, min_y, max_x, max_y = renderer.bounds(projected)
                    half_w = max(6.0, (max_x - min_x) / 2.0)
                    half_h = max(6.0, (max_y - min_y) / 2.0)
                    contact_radius = max(2.0, min(half_w, half_h))
                else:
                    radius = orb_renderer.radius(
                        focal_length, obj.size, _avg_scale(state.scale)
                    )
                    half_w = max(6.0, radius)
                    half_h = max(6.0, radius)
                    contact_radius = max(2.0, radius)

                contact_infos = hand_input.contact_vectors(hands, state.position)
                if contact_radius > 0:
                    inflated_infos = []
                    for info in contact_infos:
                        if info is None:
                            inflated_infos.append(None)
                        else:
                            distance, normal = info
                            inflated_infos.append((distance - contact_radius, normal))
                    contact_infos = inflated_infos

                contact = physics.step(state, hands, contact_infos, dt)
                contact_any = contact_any or contact
                contact_flags.append(contact)

                if obj.kind == "cube":
                    projected, _ = renderer.project(
                        (int(state.position[0]), int(state.position[1])),
                        state.yaw,
                        state.pitch,
                        focal_length,
                        state.scale,
                    )
                    min_x, min_y, max_x, max_y = renderer.bounds(projected)
                    half_w = max(6.0, (max_x - min_x) / 2.0)
                    half_h = max(6.0, (max_y - min_y) / 2.0)
                else:
                    radius = orb_renderer.radius(
                        focal_length, obj.size, _avg_scale(state.scale)
                    )
                    half_w = max(6.0, radius)
                    half_h = max(6.0, radius)
                physics.apply_bounds(state, width, height, half_w, half_h)
                radii.append(max(half_w, half_h))
                object_states.append(state)

            if len(object_states) > 1:
                _resolve_cube_collisions(object_states, radii, physics.config.restitution)

            for obj, contact in zip(objects, contact_flags):
                state = obj.state
                if state.position is None:
                    continue
                if obj.kind == "cube":
                    projected, rotated = renderer.project(
                        (int(state.position[0]), int(state.position[1])),
                        state.yaw,
                        state.pitch,
                        focal_length,
                        state.scale,
                    )
                    min_x, min_y, max_x, max_y = renderer.bounds(projected)
                    half_w = max(6.0, (max_x - min_x) / 2.0)
                    half_h = max(6.0, (max_y - min_y) / 2.0)
                    physics.apply_bounds(state, width, height, half_w, half_h)
                    projected, rotated = renderer.project(
                        (int(state.position[0]), int(state.position[1])),
                        state.yaw,
                        state.pitch,
                        focal_length,
                        state.scale,
                    )
                    renderer.draw(frame, projected, rotated, contact)
                else:
                    radius = orb_renderer.radius(
                        focal_length, obj.size, _avg_scale(state.scale)
                    )
                    physics.apply_bounds(state, width, height, radius, radius)
                    orb_renderer.draw(
                        frame,
                        (int(state.position[0]), int(state.position[1])),
                        focal_length,
                        obj.size,
                        _avg_scale(state.scale),
                        contact,
                    )

            if objects:
                avg_speed = sum(
                    math.hypot(obj.state.velocity[0], obj.state.velocity[1])
                    for obj in objects
                ) / len(objects)
                speed_history.append(avg_speed)

            mask = hand_input.stylized_mask(frame.shape)
            if mask is not None:
                frame[mask == 255] = base_frame[mask == 255]

            hand_input.draw(frame)
            menu.draw(frame)
            _draw_top_bar(frame, menu.is_open, menu_hover)
            if args.show_gestures:
                _draw_gesture_labels(
                    frame,
                    hands,
                    gesture_results,
                    args.gesture_min_confidence,
                )
            if logger is not None:
                log_mask = None
                if args.log_mask:
                    log_mask = hand_input.last_mask()
                    if log_mask is None:
                        log_mask = mask
                    if log_mask is not None:
                        log_mask = log_mask.copy()
                logger.log(
                    frame_id=frame_id,
                    timestamp=now,
                    fps=fps_value,
                    hands=hands,
                    gestures=gesture_results,
                    cubes=objects,
                    contact_flags=contact_flags,
                    frame=base_frame if args.log_frames else None,
                    overlay=frame if args.log_overlay else None,
                    mask=log_mask,
                )
            if ui_flags["stats"]:
                cube_count = _count_objects(objects, "cube")
                orb_count = _count_objects(objects, "orb")
                stats_lines = [
                    f"FPS {fps_value:.1f}",
                    f"Hands {len(hands)}",
                    f"Cubes {cube_count} Orbs {orb_count}",
                    f"Gravity {physics.config.gravity:.0f}",
                    f"Contact {'YES' if contact_any else 'NO'}",
                ]
                _draw_stats(frame, stats_lines, anchor=(max(18, width - 260), 18))
            if ui_flags["graphs"]:
                _draw_graph(frame, list(speed_history), "Speed", anchor=(max(18, width - 280), max(220, height - 180)))

            cv2.imshow("Hand Cube", frame)
            if args.show_mask and args.input_mode in ("blue-glove", "hybrid"):
                mask = hand_input.last_mask()
                if mask is not None:
                    cv2.imshow("Glove Mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if hand_worker is not None:
            hand_worker.stop()
        hand_input.close()
        if capture is not None:
            capture.release()
        else:
            cap.release()
        if logger is not None:
            logger.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
