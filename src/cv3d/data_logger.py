from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

    from .gesture_model import GestureResult
    from .hand_input import HandState
    from .physics import CubeState


@dataclass
class LoggerConfig:
    root_dir: Path
    record_frames: bool = False
    record_overlay: bool = False
    record_mask: bool = False
    every: int = 1
    jpeg_quality: int = 90
    queue_size: int = 256
    session_name: Optional[str] = None


class DataLogger:
    def __init__(self, config: LoggerConfig, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=config.queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._dropped = 0

        session = config.session_name or time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = config.root_dir / session
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.session_dir / "frames"
        self.overlay_dir = self.session_dir / "overlay"
        self.masks_dir = self.session_dir / "masks"
        if config.record_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        if config.record_overlay:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)
        if config.record_mask:
            self.masks_dir.mkdir(parents=True, exist_ok=True)
        self._data_file = (self.session_dir / "data.jsonl").open("w", encoding="utf-8")
        meta = metadata or {}
        meta["logger"] = {
            "record_frames": config.record_frames,
            "record_overlay": config.record_overlay,
            "record_mask": config.record_mask,
            "every": config.every,
            "jpeg_quality": config.jpeg_quality,
            "queue_size": config.queue_size,
        }
        safe_meta = _json_safe(meta)
        (self.session_dir / "meta.json").write_text(json.dumps(safe_meta, indent=2), encoding="utf-8")

    def start(self) -> "DataLogger":
        self._thread.start()
        return self

    def log(
        self,
        frame_id: int,
        timestamp: float,
        fps: float,
        hands: Iterable["HandState"],
        gestures: Optional[Dict[int, "GestureResult"]],
        cubes: Iterable["CubeState"],
        contact_flags: Optional[List[bool]],
        frame: Optional["np.ndarray"] = None,
        overlay: Optional["np.ndarray"] = None,
        mask: Optional["np.ndarray"] = None,
    ) -> None:
        every = max(1, int(self.config.every))
        if frame_id % every != 0:
            return
        item = {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "fps": float(fps),
            "hands": [_hand_to_dict(hand) for hand in hands],
            "gestures": _gestures_to_dict(gestures),
            "cubes": [_cube_to_dict(cube) for cube in cubes],
            "contact_flags": list(contact_flags) if contact_flags is not None else None,
        }
        if frame is not None and self.config.record_frames:
            item["frame"] = frame
        if overlay is not None and self.config.record_overlay:
            item["overlay"] = overlay
        if mask is not None and self.config.record_mask:
            item["mask"] = mask
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1

    def _loop(self) -> None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            record = {k: v for k, v in item.items() if k not in ("frame", "overlay", "mask")}
            self._data_file.write(json.dumps(record) + "\n")
            frame_id = record["frame_id"]
            if "frame" in item:
                path = self.frames_dir / f"{frame_id:06d}.jpg"
                cv2.imwrite(str(path), item["frame"], params)
            if "overlay" in item:
                path = self.overlay_dir / f"{frame_id:06d}.jpg"
                cv2.imwrite(str(path), item["overlay"], params)
            if "mask" in item:
                path = self.masks_dir / f"{frame_id:06d}.png"
                cv2.imwrite(str(path), item["mask"])

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._data_file.flush()
        self._data_file.close()
        stats = {"dropped": self._dropped}
        (self.session_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _hand_to_dict(hand: "HandState") -> Dict[str, Any]:
    return {
        "id": int(hand.id),
        "center": [float(hand.center[0]), float(hand.center[1])],
        "velocity": [float(hand.velocity[0]), float(hand.velocity[1])],
        "pinch": bool(hand.pinch),
        "pinch_strength": float(hand.pinch_strength),
        "grip": bool(hand.grip),
        "grip_strength": float(hand.grip_strength),
        "open_palm": bool(hand.open_palm),
        "open_strength": float(hand.open_strength),
        "press": bool(hand.press),
        "press_strength": float(hand.press_strength),
        "landmarks_2d": [[int(x), int(y)] for x, y in hand.landmarks_2d],
        "landmarks_3d": [[float(x), float(y), float(z)] for x, y, z in hand.landmarks_3d],
    }


def _gestures_to_dict(gestures: Optional[Dict[int, "GestureResult"]]) -> Optional[Dict[str, Any]]:
    if gestures is None:
        return None
    results: Dict[str, Any] = {}
    for hand_id, result in gestures.items():
        results[str(hand_id)] = {
            "label": result.label,
            "confidence": float(result.confidence),
            "scores": {key: float(value) for key, value in result.scores.items()},
        }
    return results


def _cube_to_dict(cube: "CubeState") -> Dict[str, Any]:
    kind = getattr(cube, "kind", "cube")
    state = getattr(cube, "state", cube)
    size = getattr(cube, "size", None)
    data = {
        "kind": str(kind),
        "position": [float(state.position[0]), float(state.position[1])] if state.position else None,
        "velocity": [float(state.velocity[0]), float(state.velocity[1])],
        "yaw": float(state.yaw),
        "pitch": float(state.pitch),
        "scale": [float(state.scale[0]), float(state.scale[1]), float(state.scale[2])],
        "grabbed_by": int(state.grabbed_by) if state.grabbed_by is not None else None,
    }
    if size is not None:
        data["size"] = float(size)
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
