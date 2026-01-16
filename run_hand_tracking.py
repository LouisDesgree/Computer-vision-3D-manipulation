#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cv3d.hand_tracking import HandTracker
from cv3d.palette import IOS_BLUE
from cv3d.pipeline import ThreadedCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand tracking demo with MediaPipe.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("auto", "any", "avfoundation"),
        default="auto",
        help="Camera backend (macOS default is AVFoundation).",
    )
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
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--flip", action="store_true", help="Mirror the camera view.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to hand_landmarker.task.",
    )
    parser.add_argument(
        "--no-threaded",
        action="store_true",
        help="Disable threaded camera capture.",
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


def _is_black_frame(frame) -> bool:
    if frame is None or frame.size == 0:
        return True
    mean = frame.mean()
    std = frame.std()
    return mean < 2.0 and std < 2.0


def _black_frame_message(index: int, backend: str) -> str:
    return (
        "Camera frames are black. On macOS this usually means camera permission is "
        "blocked for Terminal/Python or the wrong camera index is selected. "
        "Check System Settings > Privacy & Security > Camera, then try "
        f"`--camera-index {index + 1}` or `--auto-camera`. "
        f"Current backend: {backend}."
    )


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
            raise RuntimeError(
                "Unable to open camera. Try a different --camera-index."
            )

    threaded = not args.no_threaded
    capture = ThreadedCapture(cap).start() if threaded else None

    tracker = HandTracker(
        max_num_hands=args.max_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_path=args.model_path,
    )

    last_time = time.time()
    frame_count = 0
    fps = 0.0
    black_frames = 0
    max_black_frames = 60

    try:
        while True:
            if capture is not None:
                ok, frame, _frame_id = capture.read()
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if _is_black_frame(frame):
                black_frames += 1
            else:
                black_frames = 0

            if black_frames >= max_black_frames:
                message = _black_frame_message(camera_index, args.backend)
                print(message)
                if args.auto_camera and camera_index < args.max_camera_index:
                    camera_index += 1
                    if capture is not None:
                        capture.release()
                        capture = None
                    else:
                        cap.release()
                    cap = _open_camera(camera_index, backend)
                    if not cap.isOpened():
                        raise RuntimeError(message)
                    if threaded:
                        capture = ThreadedCapture(cap).start()
                    black_frames = 0
                    continue
                raise RuntimeError(message)

            if args.flip:
                frame = cv2.flip(frame, 1)

            results = tracker.process(frame)
            tracker.draw(frame, results)

            frame_count += 1
            now = time.time()
            elapsed = now - last_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_time = now

            label = f"hands: {len(results)}  fps: {fps:.1f}"
            cv2.putText(
                frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                IOS_BLUE,
                2,
            )

            cv2.imshow("Hand Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        tracker.close()
        if capture is not None:
            capture.release()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
