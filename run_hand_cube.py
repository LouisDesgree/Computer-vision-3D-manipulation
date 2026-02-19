#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
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
from cv3d.fretboard import FretboardConfig, FretboardTracker, draw_fretboard
from cv3d.gesture_model import GestureConfig, GestureTracker
from cv3d.hand_input import HandInput
from cv3d.hand_menu import HandMenu, MenuItem
from cv3d.orb_render import OrbRenderer
from cv3d.palette import (
    CUBE_BLUE,
    CUBE_RED,
    IOS_BORDER,
    IOS_BLUE,
    IOS_BLUE_SOFT,
    IOS_GLASS,
    IOS_SEPARATOR,
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


def _settings_path(custom: Path | None) -> Path:
    if custom is not None:
        return custom.expanduser()
    return Path.home() / ".cv3d" / "hand_cube_settings.json"


def _load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: failed to read settings from {path}: {exc}")
        return {}


def _save_settings(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"Warning: failed to save settings to {path}: {exc}")


def _apply_settings(
    settings: dict,
    hand_input: HandInput,
    physics: CubePhysics,
    objects: list["SceneObject"],
    ui_flags: dict[str, bool],
    mask_flags: dict[str, float | bool],
    fretboard_tracker: FretboardTracker | None,
    args: argparse.Namespace,
) -> None:
    controls = settings.get("controls", {})
    gravity = controls.get("gravity")
    if gravity is not None:
        physics.config.gravity = max(0.0, float(gravity))
    contact_force = controls.get("contact_force")
    if contact_force is not None:
        physics.config.contact_force = max(0.0, float(contact_force))
    pinch_ratio = controls.get("pinch_ratio")
    if pinch_ratio is not None:
        hand_input.set_pinch_ratio(float(pinch_ratio))
    grip_ratio = controls.get("grip_ratio")
    if grip_ratio is not None:
        hand_input.set_grip_ratio(float(grip_ratio))
    cube_count = controls.get("cubes")
    if cube_count is not None:
        _set_cube_count(objects, float(cube_count), args.cube_size)
    orb_count = controls.get("orbs")
    if orb_count is not None:
        _set_orb_count(objects, float(orb_count), args.orb_size, args.max_orbs)

    ui = settings.get("ui", {})
    for key in ("stats", "graphs", "manipulation", "mask_view"):
        if key in ui:
            ui_flags[key] = bool(ui[key])

    mask = settings.get("mask", {})
    hsv_lower = mask.get("hsv_lower")
    hsv_upper = mask.get("hsv_upper")
    if isinstance(hsv_lower, (list, tuple)) and len(hsv_lower) == 3:
        hand_input.set_glove_h_low(hsv_lower[0])
        hand_input.set_glove_s_low(hsv_lower[1])
        hand_input.set_glove_v_low(hsv_lower[2])
    if isinstance(hsv_upper, (list, tuple)) and len(hsv_upper) == 3:
        hand_input.set_glove_h_high(hsv_upper[0])
        hand_input.set_glove_s_high(hsv_upper[1])
        hand_input.set_glove_v_high(hsv_upper[2])
    min_area = mask.get("min_area")
    if min_area is not None:
        hand_input.set_glove_min_area(float(min_area))
    kernel_size = mask.get("kernel_size")
    if kernel_size is not None:
        hand_input.set_glove_kernel_size(float(kernel_size))
    morph_open = mask.get("open")
    if morph_open is not None:
        hand_input.set_glove_open(float(morph_open))
    morph_close = mask.get("close")
    if morph_close is not None:
        hand_input.set_glove_close(float(morph_close))
    morph_dilate = mask.get("dilate")
    if morph_dilate is not None:
        hand_input.set_glove_dilate(float(morph_dilate))
    epsilon = mask.get("contour_epsilon")
    if epsilon is not None:
        hand_input.set_glove_contour_epsilon(float(epsilon))
    hand_overlay = mask.get("hand_overlay")
    if hand_overlay is None and "overlay" in mask:
        hand_overlay = mask.get("overlay")
    if hand_overlay is not None:
        mask_flags["hand_overlay"] = bool(hand_overlay)
    hand_window = mask.get("hand_window")
    if hand_window is None and "window" in mask:
        hand_window = mask.get("window")
    if hand_window is not None:
        mask_flags["hand_window"] = bool(hand_window)
    for key in ("fret_overlay", "fret_window", "glove", "stylized"):
        if key in mask:
            mask_flags[key] = bool(mask[key])
    alpha = mask.get("alpha")
    if alpha is not None:
        mask_flags["alpha"] = max(0.05, min(0.85, float(alpha)))

    if fretboard_tracker is not None and args.fretboard_config is None:
        fb = settings.get("fretboard_mask", {})
        config = fretboard_tracker.config
        use_color = fb.get("use_color")
        if use_color is not None:
            config.mask_use_color = bool(use_color)
        use_depth = fb.get("use_depth")
        if use_depth is not None:
            config.mask_use_depth = bool(use_depth)
        exclude_hands = fb.get("exclude_hands")
        if exclude_hands is not None:
            config.mask_exclude_hands = bool(exclude_hands)
        color_lower = fb.get("color_lower")
        color_upper = fb.get("color_upper")
        if isinstance(color_lower, (list, tuple)) and len(color_lower) == 3:
            config.mask_color_lower = tuple(int(v) for v in color_lower)
        if isinstance(color_upper, (list, tuple)) and len(color_upper) == 3:
            config.mask_color_upper = tuple(int(v) for v in color_upper)
        for key, attr in (
            ("color_open", "mask_color_open"),
            ("color_close", "mask_color_close"),
            ("color_dilate", "mask_color_dilate"),
            ("depth_blur", "mask_depth_blur"),
            ("depth_threshold", "mask_depth_threshold"),
            ("depth_dilate", "mask_depth_dilate"),
            ("hand_dilate", "mask_hand_dilate"),
        ):
            value = fb.get(key)
            if value is not None:
                try:
                    setattr(config, attr, int(round(float(value))))
                except (TypeError, ValueError):
                    continue


def _clamp_value(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _smooth_value(current: float | None, target: float, alpha: float = 0.25) -> float:
    if current is None:
        return target
    return current + (target - current) * alpha


def _collect_settings(
    hand_input: HandInput,
    physics: CubePhysics,
    objects: list["SceneObject"],
    ui_flags: dict[str, bool],
    mask_flags: dict[str, float | bool],
    fretboard_tracker: FretboardTracker | None,
) -> dict:
    data = {
        "version": 1,
        "controls": {
            "gravity": float(physics.config.gravity),
            "contact_force": float(physics.config.contact_force),
            "cubes": int(_count_objects(objects, "cube")),
            "orbs": int(_count_objects(objects, "orb")),
            "pinch_ratio": float(hand_input.get_pinch_ratio()),
            "grip_ratio": float(hand_input.get_grip_ratio()),
        },
        "ui": {
            "stats": bool(ui_flags.get("stats", False)),
            "graphs": bool(ui_flags.get("graphs", False)),
            "fretboard": bool(ui_flags.get("fretboard", False)),
            "manipulation": bool(ui_flags.get("manipulation", True)),
            "mask_view": bool(ui_flags.get("mask_view", False)),
        },
        "mask": {
            "hsv_lower": [
                int(hand_input.get_glove_h_low()),
                int(hand_input.get_glove_s_low()),
                int(hand_input.get_glove_v_low()),
            ],
            "hsv_upper": [
                int(hand_input.get_glove_h_high()),
                int(hand_input.get_glove_s_high()),
                int(hand_input.get_glove_v_high()),
            ],
            "min_area": float(hand_input.get_glove_min_area()),
            "kernel_size": int(hand_input.get_glove_kernel_size()),
            "open": int(hand_input.get_glove_open()),
            "close": int(hand_input.get_glove_close()),
            "dilate": int(hand_input.get_glove_dilate()),
            "contour_epsilon": float(hand_input.get_glove_contour_epsilon()),
            "hand_overlay": bool(mask_flags.get("hand_overlay", False)),
            "hand_window": bool(mask_flags.get("hand_window", False)),
            "fret_overlay": bool(mask_flags.get("fret_overlay", False)),
            "fret_window": bool(mask_flags.get("fret_window", False)),
            "glove": bool(mask_flags.get("glove", True)),
            "stylized": bool(mask_flags.get("stylized", False)),
            "alpha": float(mask_flags.get("alpha", 0.35)),
        },
    }
    if fretboard_tracker is not None:
        config = fretboard_tracker.config
        data["fretboard_mask"] = {
            "use_color": bool(config.mask_use_color),
            "use_depth": bool(config.mask_use_depth),
            "exclude_hands": bool(config.mask_exclude_hands),
            "color_lower": [
                int(config.mask_color_lower[0]),
                int(config.mask_color_lower[1]),
                int(config.mask_color_lower[2]),
            ],
            "color_upper": [
                int(config.mask_color_upper[0]),
                int(config.mask_color_upper[1]),
                int(config.mask_color_upper[2]),
            ],
            "color_open": int(config.mask_color_open),
            "color_close": int(config.mask_color_close),
            "color_dilate": int(config.mask_color_dilate),
            "depth_blur": int(config.mask_depth_blur),
            "depth_threshold": int(config.mask_depth_threshold),
            "depth_dilate": int(config.mask_depth_dilate),
            "hand_dilate": int(config.mask_hand_dilate),
        }
    return data


def _dataset_path(custom: Path | None) -> Path:
    if custom is not None:
        return custom.expanduser()
    return Path.home() / ".cv3d" / "fretboard_dataset"


def _load_fretboard_overrides(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"Warning: failed to read fretboard config from {path}: {exc}")
        return {}
    except json.JSONDecodeError as exc:
        print(f"Warning: invalid fretboard config JSON in {path}: {exc}")
        return {}


def _apply_fretboard_overrides(config: FretboardConfig, overrides: dict) -> None:
    for key, value in overrides.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        try:
            if isinstance(current, bool):
                cast_value = bool(value)
            elif isinstance(current, int):
                cast_value = int(round(float(value)))
            elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
                cast_value = tuple(int(round(float(item))) for item in value)
            else:
                cast_value = float(value)
        except (TypeError, ValueError):
            continue
        setattr(config, key, cast_value)


def _next_sample_id(images_dir: Path) -> int:
    if not images_dir.exists():
        return 1
    highest = 0
    for path in images_dir.glob("*.jpg"):
        if path.stem.isdigit():
            highest = max(highest, int(path.stem))
    for path in images_dir.glob("*.png"):
        if path.stem.isdigit():
            highest = max(highest, int(path.stem))
    return highest + 1


def _count_annotations(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in path.read_text(encoding="utf-8").splitlines() if _)
    except OSError:
        return 0


def _order_quad(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(points) != 4:
        return points
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    return [(int(p[0]), int(p[1])) for p in ordered]


def _run_fretboard_training(
    cap: cv2.VideoCapture,
    capture: ThreadedCapture | None,
    args: argparse.Namespace,
    fretboard_tracker: FretboardTracker,
) -> None:
    dataset_dir = _dataset_path(args.fretboard_dataset)
    images_dir = dataset_dir / "images"
    annotations_path = dataset_dir / "annotations.jsonl"
    images_dir.mkdir(parents=True, exist_ok=True)
    sample_id = _next_sample_id(images_dir)
    sample_count = _count_annotations(annotations_path)

    points: list[tuple[int, int]] = []

    def _on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    window_name = "Fretboard Trainer"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        if capture is not None:
            ok, frame, _frame_id = capture.read()
        else:
            ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue

        if args.flip:
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        auto_result = fretboard_tracker.update(frame, time.time())
        if auto_result is not None:
            auto_poly = auto_result.polygon.astype(np.int32)
            cv2.polylines(display, [auto_poly], True, IOS_BORDER, 1)

        if points:
            poly = np.array(points, dtype=np.int32)
            cv2.polylines(display, [poly], len(points) == 4, IOS_BLUE, 2)
            for pt in points:
                cv2.circle(display, pt, 6, IOS_BLUE_SOFT, -1)

        info_lines = [
            "Training mode: click 4 corners of the fretboard.",
            "Keys: [a] auto  [s] save  [c] clear  [u] undo  [q] quit",
            f"Dataset: {dataset_dir}",
            f"Samples: {sample_count}",
        ]
        _draw_stats(display, info_lines, anchor=(18, 18))
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("u"), ord("z")) and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key == ord("a"):
            if auto_result is not None:
                auto_pts = [(int(p[0]), int(p[1])) for p in auto_result.polygon]
                points[:] = _order_quad(auto_pts)
        elif key == ord("s"):
            if len(points) == 4:
                ordered = _order_quad(points)
                image_name = f"{sample_id:06d}.jpg"
                image_path = images_dir / image_name
                cv2.imwrite(str(image_path), frame)
                entry = {
                    "image": f"images/{image_name}",
                    "polygon": ordered,
                    "timestamp": time.time(),
                }
                with annotations_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                sample_id += 1
                sample_count += 1
                points.clear()

    cv2.destroyWindow(window_name)


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
        "--train-fretboard",
        action="store_true",
        help="Run the fretboard training capture UI (manual annotation).",
    )
    parser.add_argument(
        "--fretboard-dataset",
        type=Path,
        default=None,
        help="Directory to store fretboard training samples.",
    )
    parser.add_argument(
        "--fretboard-config",
        type=Path,
        default=None,
        help="Optional JSON file to override fretboard detection parameters.",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=None,
        help="Optional path for settings JSON (default: ~/.cv3d/hand_cube_settings.json).",
    )
    parser.add_argument(
        "--no-settings",
        action="store_true",
        help="Disable loading/saving settings across launches.",
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
        "--show-fretboard",
        action="store_true",
        help="Overlay detected guitar fretboard and finger positions.",
    )
    parser.add_argument(
        "--fretboard-scale",
        type=float,
        default=0.6,
        help="Downscale factor for fretboard detection (0.3-1.0).",
    )
    parser.add_argument(
        "--fretboard-canny-low",
        type=int,
        default=50,
        help="Lower Canny threshold for fretboard edges.",
    )
    parser.add_argument(
        "--fretboard-canny-high",
        type=int,
        default=140,
        help="Upper Canny threshold for fretboard edges.",
    )
    parser.add_argument(
        "--fretboard-min-area",
        type=float,
        default=0.03,
        help="Minimum fretboard area ratio relative to the frame.",
    )
    parser.add_argument(
        "--fretboard-min-aspect",
        type=float,
        default=3.0,
        help="Minimum aspect ratio for the fretboard candidate.",
    )
    parser.add_argument(
        "--fretboard-angle-tol",
        type=float,
        default=12.0,
        help="Angle tolerance in degrees for string/fret lines.",
    )
    parser.add_argument(
        "--fretboard-strings",
        type=int,
        default=6,
        help="Expected number of strings on the fretboard.",
    )
    parser.add_argument(
        "--fretboard-line-length",
        type=int,
        default=40,
        help="Minimum line length for Hough line detection.",
    )
    parser.add_argument(
        "--fretboard-line-gap",
        type=int,
        default=10,
        help="Maximum line gap for Hough line detection.",
    )
    parser.add_argument(
        "--fretboard-string-cluster",
        type=float,
        default=0.04,
        help="String clustering tolerance as a fraction of board width.",
    )
    parser.add_argument(
        "--fretboard-fret-cluster",
        type=float,
        default=0.025,
        help="Fret clustering tolerance as a fraction of board length.",
    )
    parser.add_argument(
        "--fretboard-mask-color",
        action="store_true",
        help="Enable color mask for fretboard detection.",
    )
    parser.add_argument(
        "--fretboard-mask-color-low",
        type=_parse_hsv,
        default=(5, 30, 40),
        help="HSV lower bound for fretboard color mask.",
    )
    parser.add_argument(
        "--fretboard-mask-color-high",
        type=_parse_hsv,
        default=(30, 255, 255),
        help="HSV upper bound for fretboard color mask.",
    )
    parser.add_argument(
        "--fretboard-mask-color-open",
        type=int,
        default=0,
        help="Open iterations for fretboard color mask.",
    )
    parser.add_argument(
        "--fretboard-mask-color-close",
        type=int,
        default=1,
        help="Close iterations for fretboard color mask.",
    )
    parser.add_argument(
        "--fretboard-mask-color-dilate",
        type=int,
        default=1,
        help="Dilate iterations for fretboard color mask.",
    )
    parser.add_argument(
        "--fretboard-mask-depth",
        action="store_true",
        help="Enable depth-like mask for fretboard detection.",
    )
    parser.add_argument(
        "--fretboard-mask-depth-blur",
        type=int,
        default=5,
        help="Blur kernel size for depth-like mask.",
    )
    parser.add_argument(
        "--fretboard-mask-depth-threshold",
        type=int,
        default=18,
        help="Threshold for depth-like mask edge strength.",
    )
    parser.add_argument(
        "--fretboard-mask-depth-dilate",
        type=int,
        default=1,
        help="Dilate iterations for depth-like mask.",
    )
    parser.add_argument(
        "--fretboard-mask-exclude-hands",
        action="store_true",
        help="Exclude hand regions from fretboard detection.",
    )
    parser.add_argument(
        "--fretboard-mask-hand-dilate",
        type=int,
        default=18,
        help="Dilate iterations for the hand exclusion mask.",
    )
    parser.add_argument(
        "--fretboard-smooth",
        type=float,
        default=0.5,
        help="Smoothing factor for fretboard tracking (0-1).",
    )
    parser.add_argument(
        "--fretboard-hold",
        type=float,
        default=0.7,
        help="Seconds to keep the last fretboard if detection drops.",
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
    parser.add_argument("--num-cubes", type=int, default=1)
    parser.add_argument("--cube-size", type=float, default=0.7)
    parser.add_argument("--cube-distance", type=float, default=5.8)
    parser.add_argument("--num-orbs", type=int, default=0)
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
        default=0.0,
        help="Cooldown between two-finger spawns (0 disables).",
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


def _apply_mask_overlay(frame, mask, color, alpha: float) -> None:
    if mask is None:
        return
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.0:
        return
    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


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
    spawn_enabled = config.spawn_cooldown > 0.0 and config.max_orbs > 0
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

        if spawn_enabled:
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

    if spawn_enabled and spawn_requests > 0:
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

    window_name = "Hand Cube"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

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

    fretboard_overrides = _load_fretboard_overrides(args.fretboard_config)
    fretboard_config = FretboardConfig(
        scale=args.fretboard_scale,
        canny_low=args.fretboard_canny_low,
        canny_high=args.fretboard_canny_high,
        min_area_ratio=args.fretboard_min_area,
        min_aspect=args.fretboard_min_aspect,
        angle_tol=args.fretboard_angle_tol,
        min_line_length=args.fretboard_line_length,
        max_line_gap=args.fretboard_line_gap,
        string_count=args.fretboard_strings,
        string_cluster_ratio=args.fretboard_string_cluster,
        fret_cluster_ratio=args.fretboard_fret_cluster,
        smooth_alpha=args.fretboard_smooth,
        hold_seconds=args.fretboard_hold,
        mask_use_color=args.fretboard_mask_color,
        mask_color_lower=args.fretboard_mask_color_low,
        mask_color_upper=args.fretboard_mask_color_high,
        mask_color_open=args.fretboard_mask_color_open,
        mask_color_close=args.fretboard_mask_color_close,
        mask_color_dilate=args.fretboard_mask_color_dilate,
        mask_use_depth=args.fretboard_mask_depth,
        mask_depth_blur=args.fretboard_mask_depth_blur,
        mask_depth_threshold=args.fretboard_mask_depth_threshold,
        mask_depth_dilate=args.fretboard_mask_depth_dilate,
        mask_exclude_hands=args.fretboard_mask_exclude_hands,
        mask_hand_dilate=args.fretboard_mask_hand_dilate,
    )
    if fretboard_overrides:
        _apply_fretboard_overrides(fretboard_config, fretboard_overrides)
    fretboard_tracker = FretboardTracker(fretboard_config)

    capture = ThreadedCapture(cap).start() if threaded else None
    if args.train_fretboard:
        try:
            _run_fretboard_training(cap, capture, args, fretboard_tracker)
        finally:
            if capture is not None:
                capture.release()
            else:
                cap.release()
            cv2.destroyAllWindows()
        return

    settings_path = _settings_path(args.settings_path)
    settings = {} if args.no_settings else _load_settings(settings_path)

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
    paddle_centers: dict[str, float | None] = {"left": None, "right": None}
    clock_state = CubeState()
    ui_flags = {
        "stats": False,
        "graphs": False,
        "fretboard": args.show_fretboard,
        "manipulation": True,
        "mask_view": False,
        "mask_hsv": False,
        "mask_morph": False,
    }
    mask_flags = {
        "hand_overlay": False,
        "hand_window": bool(args.show_mask),
        "fret_overlay": False,
        "fret_window": False,
        "glove": True,
        "stylized": False,
        "alpha": 0.35,
    }

    if settings:
        _apply_settings(
            settings,
            hand_input,
            physics,
            objects,
            ui_flags,
            mask_flags,
            fretboard_tracker,
            args,
        )
        if args.show_mask:
            mask_flags["hand_window"] = True
    ui_flags["fretboard"] = bool(args.show_fretboard)
    speed_history = deque(maxlen=120)
    menu_toggle_time = 0.0
    menu_hover = False
    fps_value = 0.0
    gesture_state: dict[int, dict[str, float | str]] = {}
    hand_mask_window_open = False
    fret_mask_window_open = False
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

    def _toggle_fretboard() -> None:
        ui_flags["fretboard"] = not ui_flags["fretboard"]

    def _toggle_manipulation() -> None:
        ui_flags["manipulation"] = not ui_flags["manipulation"]

    def _toggle_mask_view() -> None:
        ui_flags["mask_view"] = not ui_flags["mask_view"]
        if not ui_flags["mask_view"]:
            ui_flags["mask_hsv"] = False
            ui_flags["mask_morph"] = False

    def _toggle_mask_hsv() -> None:
        ui_flags["mask_hsv"] = not ui_flags["mask_hsv"]

    def _toggle_mask_morph() -> None:
        ui_flags["mask_morph"] = not ui_flags["mask_morph"]

    def _toggle_hand_overlay() -> None:
        mask_flags["hand_overlay"] = not mask_flags["hand_overlay"]

    def _toggle_hand_window() -> None:
        mask_flags["hand_window"] = not mask_flags["hand_window"]

    def _toggle_fret_overlay() -> None:
        mask_flags["fret_overlay"] = not mask_flags["fret_overlay"]

    def _toggle_fret_window() -> None:
        mask_flags["fret_window"] = not mask_flags["fret_window"]

    def _toggle_mask_glove() -> None:
        mask_flags["glove"] = not mask_flags["glove"]

    def _toggle_mask_stylized() -> None:
        mask_flags["stylized"] = not mask_flags["stylized"]

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
                "Masks",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_mask_view,
                state=lambda: ui_flags["mask_view"],
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
            MenuItem(
                "Guitar",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_fretboard,
                state=lambda: ui_flags["fretboard"],
            ),
            MenuItem(
                "Manipulate",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_manipulation,
                state=lambda: ui_flags["manipulation"],
            ),
        ],
        open_hold=args.menu_open_hold,
        toggle_cooldown=1.0,
        open_strength_threshold=args.menu_open_strength,
        scale=args.menu_scale,
        use_open_gesture=False,
    )

    mask_menu_scale = max(0.85, min(args.menu_scale * 0.7, 1.4))
    mask_view_menu = HandMenu(
        [
            MenuItem(
                "Hand Overlay",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_hand_overlay,
                state=lambda: mask_flags["hand_overlay"],
            ),
            MenuItem(
                "Hand Window",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_hand_window,
                state=lambda: mask_flags["hand_window"],
            ),
            MenuItem(
                "Hand Glove",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_mask_glove,
                state=lambda: mask_flags["glove"],
            ),
            MenuItem(
                "Hand Stylized",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_mask_stylized,
                state=lambda: mask_flags["stylized"],
            ),
            MenuItem(
                "Fret Overlay",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_fret_overlay,
                state=lambda: mask_flags["fret_overlay"],
            ),
            MenuItem(
                "Fret Window",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_fret_window,
                state=lambda: mask_flags["fret_window"],
            ),
            MenuItem(
                "Alpha",
                lambda: float(mask_flags["alpha"]),
                lambda value: mask_flags.__setitem__(
                    "alpha", max(0.05, min(0.85, float(value)))
                ),
                0.05,
                0.85,
                0.05,
                "{:.2f}",
            ),
            MenuItem(
                "HSV Menu",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_mask_hsv,
                state=lambda: ui_flags["mask_hsv"],
            ),
            MenuItem(
                "Morph Menu",
                lambda: 0.0,
                lambda _value: None,
                0.0,
                1.0,
                0.0,
                "{:.0f}",
                kind="button",
                on_press=_toggle_mask_morph,
                state=lambda: ui_flags["mask_morph"],
            ),
        ],
        anchor=(24, 24),
        width=300,
        row_height=38,
        padding=12,
        title="Mask View",
        scale=mask_menu_scale,
        use_open_gesture=False,
    )

    mask_hsv_menu = HandMenu(
        [
            MenuItem(
                "H low",
                hand_input.get_glove_h_low,
                hand_input.set_glove_h_low,
                0.0,
                179.0,
                1.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "H high",
                hand_input.get_glove_h_high,
                hand_input.set_glove_h_high,
                0.0,
                179.0,
                1.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "S low",
                hand_input.get_glove_s_low,
                hand_input.set_glove_s_low,
                0.0,
                255.0,
                5.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "S high",
                hand_input.get_glove_s_high,
                hand_input.set_glove_s_high,
                0.0,
                255.0,
                5.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "V low",
                hand_input.get_glove_v_low,
                hand_input.set_glove_v_low,
                0.0,
                255.0,
                5.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "V high",
                hand_input.get_glove_v_high,
                hand_input.set_glove_v_high,
                0.0,
                255.0,
                5.0,
                "{:.0f}",
                integer=True,
            ),
        ],
        anchor=(24, 24),
        width=280,
        row_height=36,
        padding=12,
        title="Mask HSV",
        scale=mask_menu_scale,
        use_open_gesture=False,
    )

    mask_morph_menu = HandMenu(
        [
            MenuItem(
                "Min area",
                hand_input.get_glove_min_area,
                hand_input.set_glove_min_area,
                200.0,
                8000.0,
                100.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "Kernel",
                hand_input.get_glove_kernel_size,
                hand_input.set_glove_kernel_size,
                1.0,
                15.0,
                2.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "Open",
                hand_input.get_glove_open,
                hand_input.set_glove_open,
                0.0,
                4.0,
                1.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "Close",
                hand_input.get_glove_close,
                hand_input.set_glove_close,
                0.0,
                4.0,
                1.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "Dilate",
                hand_input.get_glove_dilate,
                hand_input.set_glove_dilate,
                0.0,
                4.0,
                1.0,
                "{:.0f}",
                integer=True,
            ),
            MenuItem(
                "Epsilon",
                hand_input.get_glove_contour_epsilon,
                hand_input.set_glove_contour_epsilon,
                0.0,
                0.03,
                0.002,
                "{:.3f}",
            ),
        ],
        anchor=(24, 24),
        width=300,
        row_height=36,
        padding=12,
        title="Mask Morph",
        scale=mask_menu_scale,
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
            divider_x = width // 2
            cv2.line(frame, (divider_x, 0), (divider_x, height), IOS_SEPARATOR, 2)

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

            mask_view_menu.is_open = ui_flags["mask_view"]
            mask_hsv_menu.is_open = ui_flags["mask_hsv"]
            mask_morph_menu.is_open = ui_flags["mask_morph"]

            anchor_x, anchor_y = 24, 24
            if menu.is_open:
                panel_x, panel_y, panel_w, _panel_h = menu._panel_rect(frame.shape)
                anchor_x = panel_x + panel_w + 16
                anchor_y = panel_y
            mask_view_menu.anchor = [anchor_x, anchor_y]
            if mask_view_menu.is_open:
                panel_x, panel_y, panel_w, _panel_h = mask_view_menu._panel_rect(frame.shape)
                anchor_x = panel_x + panel_w + 16
                anchor_y = panel_y
            mask_hsv_menu.anchor = [anchor_x, anchor_y]
            if mask_hsv_menu.is_open:
                panel_x, panel_y, panel_w, _panel_h = mask_hsv_menu._panel_rect(frame.shape)
                anchor_x = panel_x + panel_w + 16
                anchor_y = panel_y
            mask_morph_menu.anchor = [anchor_x, anchor_y]

            mask_view_menu.update(hands, now, frame.shape)
            mask_hsv_menu.update(hands, now, frame.shape)
            mask_morph_menu.update(hands, now, frame.shape)
            mask_view_menu.handle_input(hands, frame.shape, now)
            mask_hsv_menu.handle_input(hands, frame.shape, now)
            mask_morph_menu.handle_input(hands, frame.shape, now)

            paddle_height = max(60, int(height * 0.18))
            paddle_width = max(12, int(width * 0.02))
            paddle_margin = int(max(24, width * 0.06))
            if paddle_centers["left"] is None:
                paddle_centers["left"] = height * 0.5
                paddle_centers["right"] = height * 0.5
            left_candidates = [
                center[1]
                for hand in hands
                if (center := getattr(hand, "center", None)) is not None and center[0] < divider_x
            ]
            right_candidates = [
                center[1]
                for hand in hands
                if (center := getattr(hand, "center", None)) is not None and center[0] >= divider_x
            ]
            left_target = (
                sum(left_candidates) / len(left_candidates)
                if left_candidates
                else (paddle_centers["left"] or height * 0.5)
            )
            right_target = (
                sum(right_candidates) / len(right_candidates)
                if right_candidates
                else (paddle_centers["right"] or height * 0.5)
            )
            new_left = _smooth_value(paddle_centers["left"], left_target, 0.25)
            new_right = _smooth_value(paddle_centers["right"], right_target, 0.25)
            paddle_centers["left"] = _clamp_value(new_left, paddle_height, height - paddle_height)
            paddle_centers["right"] = _clamp_value(new_right, paddle_height, height - paddle_height)
            left_rect = (
                int(paddle_margin - paddle_width // 2),
                int(_clamp_value(paddle_centers["left"] - paddle_height, 0, height)),
                int(paddle_margin + paddle_width // 2),
                int(_clamp_value(paddle_centers["left"] + paddle_height, 0, height)),
            )
            right_rect = (
                int(width - paddle_margin - paddle_width // 2),
                int(_clamp_value(paddle_centers["right"] - paddle_height, 0, height)),
                int(width - paddle_margin + paddle_width // 2),
                int(_clamp_value(paddle_centers["right"] + paddle_height, 0, height)),
            )
            for rect, color in ((left_rect, CUBE_BLUE), (right_rect, CUBE_RED)):
                x1, y1, x2, y2 = rect
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), IOS_BORDER, 2)

            interaction_hands = hands if ui_flags["manipulation"] else []

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
                interaction_hands,
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

                contact_infos = hand_input.contact_vectors(interaction_hands, state.position)
                if contact_radius > 0:
                    inflated_infos = []
                    for info in contact_infos:
                        if info is None:
                            inflated_infos.append(None)
                        else:
                            distance, normal = info
                            inflated_infos.append((distance - contact_radius, normal))
                    contact_infos = inflated_infos

                contact = physics.step(state, interaction_hands, contact_infos, dt)
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
                    side_color = CUBE_RED if state.position[0] > divider_x else CUBE_BLUE
                    renderer.draw(
                        frame,
                        projected,
                        rotated,
                        contact,
                        accent_color=side_color,
                    )
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

            stylized_mask = hand_input.stylized_mask(frame.shape)
            if stylized_mask is not None:
                frame[stylized_mask == 255] = base_frame[stylized_mask == 255]

            glove_mask = hand_input.last_mask()
            fret_mask = None
            if mask_flags["fret_overlay"] or mask_flags["fret_window"]:
                fret_mask = fretboard_tracker.build_mask(base_frame, hands)
            if mask_flags["hand_overlay"]:
                if mask_flags["glove"] and glove_mask is not None:
                    _apply_mask_overlay(frame, glove_mask, IOS_BLUE, mask_flags["alpha"])
                if mask_flags["stylized"] and stylized_mask is not None:
                    _apply_mask_overlay(
                        frame,
                        stylized_mask,
                        IOS_BLUE_SOFT,
                        mask_flags["alpha"] * 0.85,
                    )
            if mask_flags["fret_overlay"] and fret_mask is not None:
                _apply_mask_overlay(frame, fret_mask, IOS_BLUE_SOFT, mask_flags["alpha"])

            fretboard_result = None
            fretboard_placements = None
            if ui_flags["fretboard"]:
                fretboard_result = fretboard_tracker.update(base_frame, now, hands)
                if fretboard_result is not None:
                    fretboard_placements = fretboard_tracker.locate_fingers(
                        hands, fretboard_result
                    )
                    draw_fretboard(frame, fretboard_result, fretboard_placements)

            hand_input.draw(frame)
            menu.draw(frame)
            mask_view_menu.draw(frame)
            mask_hsv_menu.draw(frame)
            mask_morph_menu.draw(frame)
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
                        log_mask = stylized_mask
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
                    fretboard=fretboard_result,
                    finger_placements=fretboard_placements,
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

            cv2.imshow(window_name, frame)
            if mask_flags["hand_window"]:
                window_mask = None
                if mask_flags["glove"] and glove_mask is not None:
                    window_mask = glove_mask.copy()
                if mask_flags["stylized"] and stylized_mask is not None:
                    if window_mask is None:
                        window_mask = stylized_mask.copy()
                    else:
                        window_mask = np.maximum(window_mask, stylized_mask)
                if window_mask is not None:
                    cv2.imshow("Hand Mask", window_mask)
                    hand_mask_window_open = True
            elif hand_mask_window_open:
                cv2.destroyWindow("Hand Mask")
                hand_mask_window_open = False
            if mask_flags["fret_window"]:
                if fret_mask is not None:
                    cv2.imshow("Fretboard Mask", fret_mask)
                    fret_mask_window_open = True
            elif fret_mask_window_open:
                cv2.destroyWindow("Fretboard Mask")
                fret_mask_window_open = False
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if hand_worker is not None:
            hand_worker.stop()
        if not args.no_settings:
            settings = _collect_settings(
                hand_input, physics, objects, ui_flags, mask_flags, fretboard_tracker
            )
            _save_settings(settings_path, settings)
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
