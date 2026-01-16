from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np

from .palette import IOS_SHADOW


@lru_cache(maxsize=64)
def _rounded_mask(size: Tuple[int, int], radius: int) -> np.ndarray:
    height, width = size
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = max(0, min(radius, width // 2, height // 2))
    if radius == 0:
        mask[:, :] = 255
        return mask
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)
    return mask


def draw_rounded_rect(
    frame,
    rect: Tuple[int, int, int, int],
    radius: int,
    color,
    thickness: int = -1,
) -> None:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return
    radius = max(0, min(radius, w // 2, h // 2))
    if radius == 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return

    if thickness < 0:
        cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, -1)
        cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, -1)
        cv2.circle(frame, (x + radius, y + radius), radius, color, -1)
        cv2.circle(frame, (x + w - radius, y + radius), radius, color, -1)
        cv2.circle(frame, (x + radius, y + h - radius), radius, color, -1)
        cv2.circle(frame, (x + w - radius, y + h - radius), radius, color, -1)
        return

    cv2.line(frame, (x + radius, y), (x + w - radius, y), color, thickness)
    cv2.line(frame, (x + radius, y + h), (x + w - radius, y + h), color, thickness)
    cv2.line(frame, (x, y + radius), (x, y + h - radius), color, thickness)
    cv2.line(frame, (x + w, y + radius), (x + w, y + h - radius), color, thickness)
    cv2.ellipse(frame, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_shadow(
    frame,
    rect: Tuple[int, int, int, int],
    radius: int,
    alpha: float = 0.18,
    blur_sigma: float = 10.0,
    offset: Tuple[int, int] = (0, 2),
    color=IOS_SHADOW,
) -> None:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return
    pad = int(max(6, blur_sigma * 2))
    shadow = np.zeros((h + pad * 2, w + pad * 2, 3), dtype=np.uint8)
    draw_rounded_rect(
        shadow,
        (pad + offset[0], pad + offset[1], w, h),
        radius,
        color,
        -1,
    )
    shadow = cv2.GaussianBlur(shadow, (0, 0), blur_sigma)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    sx0 = x0 - (x - pad)
    sy0 = y0 - (y - pad)
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    roi = frame[y0:y1, x0:x1]
    cv2.addWeighted(shadow[sy0:sy1, sx0:sx1], alpha, roi, 1.0, 0, roi)


def apply_glass(
    frame,
    rect: Tuple[int, int, int, int],
    radius: int,
    tint_color,
    tint_alpha: float = 0.22,
    blur_sigma: float = 12.0,
) -> None:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return
    frame_h, frame_w = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + w)
    y1 = min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return
    blurred = cv2.GaussianBlur(roi, (0, 0), blur_sigma)
    tinted = blurred.copy()
    tint = np.full_like(tinted, tint_color)
    cv2.addWeighted(tint, tint_alpha, tinted, 1.0 - tint_alpha, 0, tinted)
    mask = _rounded_mask((h, w), radius)
    offset_x = x0 - x
    offset_y = y0 - y
    mask = mask[offset_y : offset_y + (y1 - y0), offset_x : offset_x + (x1 - x0)]
    roi[mask == 255] = tinted[mask == 255]


def draw_glass_panel(
    frame,
    rect: Tuple[int, int, int, int],
    radius: int,
    tint_color,
    border_color,
    border_thickness: int = 1,
    tint_alpha: float = 0.22,
    blur_sigma: float = 12.0,
    shadow: bool = True,
    shadow_alpha: float = 0.18,
) -> None:
    if shadow:
        draw_shadow(frame, rect, radius, alpha=shadow_alpha)
    apply_glass(frame, rect, radius, tint_color, tint_alpha=tint_alpha, blur_sigma=blur_sigma)
    draw_rounded_rect(frame, rect, radius, border_color, border_thickness)


def draw_text(
    frame,
    text: str,
    origin: Tuple[int, int],
    font,
    scale: float,
    color,
    thickness: int = 1,
    shadow: bool = True,
    shadow_offset: Tuple[int, int] = (0, 1),
    shadow_color=IOS_SHADOW,
) -> None:
    if shadow:
        cv2.putText(
            frame,
            text,
            (origin[0] + shadow_offset[0], origin[1] + shadow_offset[1]),
            font,
            scale,
            shadow_color,
            thickness + 1,
        )
    cv2.putText(frame, text, origin, font, scale, color, thickness)
