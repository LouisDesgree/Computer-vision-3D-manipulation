from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .palette import IOS_BLUE, IOS_BLUE_SOFT, IOS_BORDER, IOS_KNOB, IOS_SHADOW


class OrbRenderer:
    def __init__(self, distance: float) -> None:
        self.distance = max(0.1, float(distance))

    def radius(self, focal_length: float, size: float, scale: float) -> float:
        scale = max(0.2, float(scale))
        size = max(0.05, float(size))
        return max(6.0, focal_length * size * scale / self.distance)

    def draw(
        self,
        frame,
        center: Tuple[int, int],
        focal_length: float,
        size: float,
        scale: float,
        highlight: bool,
    ) -> None:
        radius = int(self.radius(focal_length, size, scale))
        if radius <= 0:
            return
        x, y = center
        glow = np.zeros_like(frame)
        glow_color = IOS_BLUE if highlight else IOS_BLUE_SOFT
        cv2.circle(glow, (x, y), int(radius * 1.4), glow_color, 2)
        cv2.addWeighted(glow, 0.18, frame, 0.82, 0, frame)

        fill = IOS_BLUE if highlight else IOS_BLUE_SOFT
        cv2.circle(frame, (x, y), radius, fill, -1)
        cv2.circle(frame, (x, y), radius, IOS_BORDER, 1)

        highlight_radius = max(2, int(radius * 0.35))
        highlight_pos = (int(x - radius * 0.3), int(y - radius * 0.35))
        cv2.circle(frame, (highlight_pos[0], highlight_pos[1] + 1), highlight_radius, IOS_SHADOW, -1)
        cv2.circle(frame, highlight_pos, highlight_radius, IOS_KNOB, -1)
