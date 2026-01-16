from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cv2

    from .hand_input import HandInput, HandState


class ThreadedCapture:
    def __init__(self, cap: "cv2.VideoCapture", sleep: float = 0.001) -> None:
        self._cap = cap
        self._sleep = max(0.0, float(sleep))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._frame: Optional[np.ndarray] = None
        self._ok = False
        self._frame_id = 0

    def start(self) -> "ThreadedCapture":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._ok = False
                time.sleep(self._sleep)
                continue
            with self._lock:
                self._frame = frame
                self._ok = True
                self._frame_id += 1

    def read(self, copy: bool = True) -> Tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return False, None, self._frame_id
            frame = self._frame.copy() if copy else self._frame
            return self._ok, frame, self._frame_id

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def release(self) -> None:
        self.stop()
        self._cap.release()


class HandWorker:
    def __init__(self, hand_input: "HandInput", sleep: float = 0.002) -> None:
        self._hand_input = hand_input
        self._sleep = max(0.0, float(sleep))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._last_frame_id = 0
        self._hands: list["HandState"] = []

    def start(self) -> "HandWorker":
        self._thread.start()
        return self

    def submit(self, frame: np.ndarray, frame_id: int) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id = frame_id

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if self._frame is None or self._frame_id == self._last_frame_id:
                    frame = None
                    frame_id = self._frame_id
                else:
                    frame = self._frame
                    frame_id = self._frame_id
            if frame is None:
                time.sleep(self._sleep)
                continue
            hands = list(self._hand_input.update(frame))
            with self._lock:
                self._hands = hands
                self._last_frame_id = frame_id

    def get(self) -> Tuple[list["HandState"], int]:
        with self._lock:
            return list(self._hands), self._last_frame_id

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
