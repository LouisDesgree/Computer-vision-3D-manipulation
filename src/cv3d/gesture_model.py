from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .hand_input import HandState


@dataclass
class GestureConfig:
    pinch_ratio: float = 0.35
    extended_ratio: float = 1.2
    curled_ratio: float = 0.9
    smoothing: float = 0.6
    min_confidence: float = 0.25
    stale_timeout: float = 0.8


@dataclass
class GestureResult:
    label: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)


FINGER_POINTS = {
    "thumb": (4, 2),
    "index": (8, 5),
    "middle": (12, 9),
    "ring": (16, 13),
    "pinky": (20, 17),
}

PALM_POINTS = (0, 5, 9, 13, 17)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class GestureModel:
    def __init__(self, config: Optional[GestureConfig] = None) -> None:
        self.config = config or GestureConfig()
        self._ema: Dict[str, float] = {
            "pinch": 0.0,
            "point": 0.0,
            "two": 0.0,
            "open": 0.0,
            "fist": 0.0,
        }

    def reset(self) -> None:
        for key in self._ema:
            self._ema[key] = 0.0

    def update(self, hand: "HandState") -> GestureResult:
        scores = self._score(hand)
        alpha = _clamp(self.config.smoothing, 0.05, 0.95)
        for label in self._ema:
            raw = scores.get(label, 0.0)
            self._ema[label] = raw * alpha + self._ema[label] * (1.0 - alpha)

        label = max(self._ema, key=self._ema.get)
        top_score = self._ema[label]
        if top_score < self.config.min_confidence:
            return GestureResult(label="unknown", confidence=top_score, scores=scores)
        return GestureResult(label=label, confidence=top_score, scores=scores)

    def _score(self, hand: "HandState") -> Dict[str, float]:
        points = hand.landmarks_2d
        if len(points) < 21:
            return {}
        palm_center = (
            sum(points[idx][0] for idx in PALM_POINTS) / len(PALM_POINTS),
            sum(points[idx][1] for idx in PALM_POINTS) / len(PALM_POINTS),
        )
        palm_size = sum(_dist(points[idx], palm_center) for idx in PALM_POINTS[1:]) / (
            len(PALM_POINTS) - 1
        )
        if palm_size < 1.0:
            return {}

        finger_ratios: Dict[str, float] = {}
        extended: Dict[str, bool] = {}
        curled: Dict[str, bool] = {}
        for name, (tip_idx, mcp_idx) in FINGER_POINTS.items():
            tip_dist = _dist(points[tip_idx], palm_center)
            mcp_dist = _dist(points[mcp_idx], palm_center)
            ratio = tip_dist / max(mcp_dist, 1.0)
            finger_ratios[name] = ratio
            extended[name] = ratio > self.config.extended_ratio
            curled[name] = ratio < self.config.curled_ratio

        thumb_tip = points[4]
        index_tip = points[8]
        pinch_dist = _dist(thumb_tip, index_tip) / palm_size
        pinch_score = _clamp(1.0 - pinch_dist / max(self.config.pinch_ratio, 1e-3), 0.0, 1.0)

        open_score = sum(1.0 if extended[name] else 0.0 for name in FINGER_POINTS) / 5.0
        fist_score = sum(1.0 if curled[name] else 0.0 for name in FINGER_POINTS) / 5.0

        point_score = (
            1.0
            if extended["index"]
            and curled["middle"]
            and curled["ring"]
            and curled["pinky"]
            else 0.0
        )
        two_score = (
            1.0
            if extended["index"]
            and extended["middle"]
            and curled["ring"]
            and curled["pinky"]
            else 0.0
        )

        scores: Dict[str, float] = {
            "pinch": pinch_score * (1.0 - 0.2 * open_score),
            "point": point_score,
            "two": two_score,
            "open": open_score,
            "fist": fist_score,
        }
        return scores


class GestureTracker:
    def __init__(self, config: Optional[GestureConfig] = None) -> None:
        self.config = config or GestureConfig()
        self._models: Dict[int, GestureModel] = {}
        self._last_seen: Dict[int, float] = {}

    def reset(self) -> None:
        self._models.clear()
        self._last_seen.clear()

    def update(self, hands: List["HandState"], now: Optional[float] = None) -> Dict[int, GestureResult]:
        if now is None:
            now = time.time()
        results: Dict[int, GestureResult] = {}
        for hand in hands:
            model = self._models.get(hand.id)
            if model is None:
                model = GestureModel(self.config)
                self._models[hand.id] = model
            results[hand.id] = model.update(hand)
            self._last_seen[hand.id] = now

        stale_ids = [
            hand_id
            for hand_id, last in self._last_seen.items()
            if now - last > self.config.stale_timeout
        ]
        for hand_id in stale_ids:
            self._models.pop(hand_id, None)
            self._last_seen.pop(hand_id, None)
        return results
