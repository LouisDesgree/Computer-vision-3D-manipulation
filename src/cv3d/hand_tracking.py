from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple
import urllib.request

import cv2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image
from mediapipe.tasks.python.vision.core.image import ImageFormat

from .palette import IOS_BLUE, IOS_KNOB

@dataclass
class HandTrackingResult:
    landmarks_2d: List[Tuple[int, int]]
    landmarks_3d: List[Tuple[float, float, float]]
    world_landmarks: Optional[List[Tuple[float, float, float]]]
    handedness: Optional[str]
    raw_landmarks: Any


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[Path] = None,
    ) -> None:
        _ = model_complexity
        self._model_path = model_path or _default_model_path()
        _ensure_model(self._model_path)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process(self, frame_bgr) -> List[HandTrackingResult]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not results.hand_landmarks:
            return []

        height, width = frame_bgr.shape[:2]
        outputs: List[HandTrackingResult] = []
        for index, hand_landmarks in enumerate(results.hand_landmarks):
            landmarks_2d = [
                (int(lm.x * width), int(lm.y * height))
                for lm in hand_landmarks
            ]
            landmarks_3d = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

            world_landmarks = None
            if results.hand_world_landmarks:
                world = results.hand_world_landmarks[index]
                world_landmarks = [(lm.x, lm.y, lm.z) for lm in world]

            handedness = None
            if results.handedness and results.handedness[index]:
                category = results.handedness[index][0]
                handedness = category.category_name or category.display_name

            outputs.append(
                HandTrackingResult(
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    world_landmarks=world_landmarks,
                    handedness=handedness,
                    raw_landmarks=hand_landmarks,
                )
            )

        return outputs

    def draw(self, frame_bgr, results: List[HandTrackingResult]) -> None:
        for result in results:
            if len(result.landmarks_2d) < 21:
                continue
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(
                    frame_bgr,
                    result.landmarks_2d[start_idx],
                    result.landmarks_2d[end_idx],
                    IOS_BLUE,
                    2,
                )
            for point in result.landmarks_2d:
                cv2.circle(frame_bgr, point, 3, IOS_KNOB, -1)

    def close(self) -> None:
        self._landmarker.close()


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _default_model_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "models" / "hand_landmarker.task"


def _ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(_model_url(), temp_path)
        temp_path.replace(model_path)
    except Exception as exc:  # pragma: no cover - defensive
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            "Failed to download hand landmarker model. "
            "Check your network connection and try again."
        ) from exc


def _model_url() -> str:
    return (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
