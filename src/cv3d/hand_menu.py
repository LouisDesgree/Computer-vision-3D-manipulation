from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, List, Optional, Tuple

import cv2

from .palette import (
    IOS_BLUE,
    IOS_BLUE_SOFT,
    IOS_BORDER,
    IOS_GLASS,
    IOS_GLASS_STRONG,
    IOS_SEPARATOR,
    IOS_SHADOW,
    IOS_TEXT,
    IOS_TEXT_MUTED,
    IOS_KNOB,
    IOS_TRACK,
)
from .ui import draw_glass_panel, draw_rounded_rect, draw_text

@dataclass
class MenuItem:
    label: str
    get_value: Callable[[], float]
    set_value: Callable[[float], None]
    min_value: float
    max_value: float
    step: float
    fmt: str = "{:.2f}"
    integer: bool = False
    kind: str = "slider"
    on_press: Optional[Callable[[], None]] = None
    state: Optional[Callable[[], bool]] = None


class HandMenu:
    def __init__(
        self,
        items: List[MenuItem],
        anchor: Tuple[int, int] = (24, 24),
        width: int = 360,
        row_height: int = 44,
        padding: int = 16,
        open_hold: float = 1.0,
        toggle_cooldown: float = 1.0,
        open_strength_threshold: float = 0.35,
        scale: float = 1.0,
        press_cooldown: float = 0.25,
        use_open_gesture: bool = True,
    ) -> None:
        scale = max(0.6, float(scale))
        self.items = items
        self.anchor = list(anchor)
        self.width = int(width * scale)
        self.row_height = int(row_height * scale)
        self.padding = int(padding * scale)
        self.open_hold = open_hold
        self.toggle_cooldown = toggle_cooldown
        self.open_strength_threshold = max(0.1, open_strength_threshold)
        self._scale = scale
        self._press_cooldown = max(0.1, press_cooldown)
        self._use_open_gesture = use_open_gesture
        self.is_open = False
        self._open_start: Optional[float] = None
        self._open_progress: float = 0.0
        self._cooldown_until: float = 0.0
        self._active_item: Optional[int] = None
        self._active_hand_id: Optional[int] = None
        self._last_pointer: Optional[Tuple[float, float]] = None
        self._last_press_time: float = 0.0
        self._last_press_hand: Optional[int] = None

    def update(self, hands, now: Optional[float] = None, frame_shape=None) -> None:
        if now is None:
            now = time.time()
        if not self._use_open_gesture:
            self._open_start = None
            self._open_progress = 0.0
            return
        if any(getattr(hand, "grip", False) or getattr(hand, "pinch", False) for hand in hands):
            self._open_start = None
            self._open_progress = 0.0
            return
        if self.is_open and frame_shape is not None and hands:
            panel = self._panel_rect(frame_shape)
            if any(self._point_in_rect(hand.center, panel) for hand in hands):
                self._open_start = None
                self._open_progress = 0.0
                return
        open_hand = next(
            (
                hand
                for hand in hands
                if getattr(hand, "open_palm", False)
                and getattr(hand, "open_strength", 0.0) >= self.open_strength_threshold
            ),
            None,
        )
        if open_hand is None:
            self._open_start = None
            self._open_progress = 0.0
            return

    def toggle(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        if now < self._cooldown_until:
            return
        self.is_open = not self.is_open
        self._cooldown_until = now + self.toggle_cooldown
        self._open_start = None
        self._open_progress = 0.0
        if self._open_start is None:
            self._open_start = now
            self._open_progress = 0.0
            return
        elapsed = now - self._open_start
        self._open_progress = min(1.0, elapsed / max(self.open_hold, 0.1))
        if elapsed >= self.open_hold and now >= self._cooldown_until:
            self.is_open = not self.is_open
            self._cooldown_until = now + self.toggle_cooldown
            self._open_start = None
            self._open_progress = 0.0

    def handle_input(self, hands, frame_shape, now: Optional[float] = None) -> None:
        if not self.is_open:
            self._active_item = None
            self._active_hand_id = None
            self._last_pointer = None
            return
        if now is None:
            now = time.time()

        panel = self._panel_rect(frame_shape)
        panel_x, panel_y, panel_w, _panel_h = panel
        panel_center = (panel_x + panel_w / 2.0, panel_y + _panel_h / 2.0)
        grab_hands = [hand for hand in hands if getattr(hand, "grip", False)]
        press_hands = [hand for hand in hands if getattr(hand, "press", False)]
        if not grab_hands and not press_hands:
            self._active_item = None
            self._active_hand_id = None
            self._last_pointer = None
            return

        if grab_hands:
            grab_hand = min(
                grab_hands,
                key=lambda hand: (hand.center[0] - panel_center[0]) ** 2
                + (hand.center[1] - panel_center[1]) ** 2,
            )
        else:
            grab_hand = min(
                press_hands,
                key=lambda hand: (hand.center[0] - panel_center[0]) ** 2
                + (hand.center[1] - panel_center[1]) ** 2,
            )
        pointer = grab_hand.center
        if getattr(grab_hand, "landmarks_2d", None) and len(grab_hand.landmarks_2d) > 8:
            pointer = grab_hand.landmarks_2d[8]
        self._last_pointer = pointer

        if self._active_hand_id is not None and grab_hand.id != self._active_hand_id:
            self._active_item = None
        self._active_hand_id = grab_hand.id

        if self._active_item is None:
            for index, item in enumerate(self.items):
                row_rect = self._row_rect(panel, index)
                if self._point_in_rect(pointer, row_rect):
                    self._active_item = index
                    break

        if self._active_item is None:
            return

        row_rect = self._row_rect(panel, self._active_item)
        if not self._point_in_rect(pointer, row_rect):
            return
        item = self.items[self._active_item]

        if item.kind == "button":
            if grab_hand not in press_hands:
                return
            if (
                self._last_press_hand == grab_hand.id
                and now - self._last_press_time < self._press_cooldown
            ):
                return
            if item.on_press is not None:
                item.on_press()
                self._last_press_time = now
                self._last_press_hand = grab_hand.id
            return

        if grab_hand in grab_hands:
            slider_left, slider_right, _slider_center = self._slider_bounds(row_rect)
            if slider_right <= slider_left:
                return
            ratio = (pointer[0] - slider_left) / float(slider_right - slider_left)
            ratio = max(0.0, min(1.0, ratio))
            value = item.min_value + ratio * (item.max_value - item.min_value)
            step = item.step
            if step > 0:
                value = round(value / step) * step
            if item.integer:
                value = int(round(value))
            item.set_value(value)
            return

        if grab_hand not in press_hands:
            return

        if (
            self._last_press_hand == grab_hand.id
            and now - self._last_press_time < self._press_cooldown
        ):
            return

        minus_rect, plus_rect = self._button_rects(row_rect)
        if self._point_in_rect(pointer, minus_rect):
            self._apply_step(self._active_item, -1.0)
        elif self._point_in_rect(pointer, plus_rect):
            self._apply_step(self._active_item, 1.0)
        else:
            return
        self._last_press_time = now
        self._last_press_hand = grab_hand.id

    def draw(self, frame) -> None:
        if not self.is_open:
            return
        panel = self._panel_rect(frame.shape)
        panel_x, panel_y, panel_w, panel_h = panel
        radius = max(12, int(18 * self._scale))
        draw_glass_panel(
            frame,
            panel,
            radius,
            IOS_GLASS,
            IOS_BORDER,
            border_thickness=max(1, int(2 * self._scale)),
            tint_alpha=0.28,
            blur_sigma=14.0,
            shadow=True,
            shadow_alpha=0.22,
        )
        title_y = panel_y + int(24 * self._scale)
        draw_text(
            frame,
            "Controls",
            (panel_x + self.padding, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.55, 0.55 * self._scale),
            IOS_TEXT,
            max(1, int(1 * self._scale)),
            shadow=False,
        )
        sep_y = panel_y + int(32 * self._scale)
        cv2.line(
            frame,
            (panel_x + self.padding, sep_y),
            (panel_x + panel_w - self.padding, sep_y),
            IOS_SEPARATOR,
            1,
        )
        for index, item in enumerate(self.items):
            row_rect = self._row_rect(panel, index)
            self._draw_item(frame, item, row_rect, index == self._active_item, index)

        if self._last_pointer is not None:
            cv2.circle(
                frame,
                (int(self._last_pointer[0]), int(self._last_pointer[1])),
                max(10, int(10 * self._scale)),
                IOS_BLUE_SOFT,
                max(2, int(2 * self._scale)),
            )
            cv2.circle(
                frame,
                (int(self._last_pointer[0]), int(self._last_pointer[1])),
                max(6, int(6 * self._scale)),
                IOS_BLUE,
                max(1, int(1 * self._scale)),
            )

    def _panel_rect(self, frame_shape) -> Tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        panel_w = min(self.width, max(220, width - 2 * self.padding))
        panel_h = self.padding * 2 + 34 + self.row_height * len(self.items)
        x = min(max(self.anchor[0], self.padding), max(self.padding, width - panel_w - self.padding))
        y = min(max(self.anchor[1], self.padding), max(self.padding, height - panel_h - self.padding))
        return int(x), int(y), int(panel_w), int(panel_h)

    def _row_rect(
        self, panel: Tuple[int, int, int, int], index: int
    ) -> Tuple[int, int, int, int]:
        panel_x, panel_y, panel_w, _panel_h = panel
        top = panel_y + 34 + self.padding + index * self.row_height
        return panel_x + self.padding, top, panel_w - self.padding * 2, self.row_height - 8

    def _slider_bounds(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        x, y, w, h = rect
        label_width = int(max(90, min(150 * self._scale, w * 0.45)))
        slider_left = x + label_width
        minus_rect, _plus_rect = self._button_rects(rect)
        slider_right = minus_rect[0] - max(6, int(6 * self._scale))
        slider_center_y = y + h // 2
        return slider_left, slider_right, slider_center_y

    def _button_rects(
        self, rect: Tuple[int, int, int, int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        x, y, w, h = rect
        size = int(min(h, max(20, 32 * self._scale)))
        spacing = int(max(4, 6 * self._scale))
        top = y + (h - size) // 2
        plus_rect = (x + w - size, top, size, size)
        minus_rect = (x + w - size * 2 - spacing, top, size, size)
        return minus_rect, plus_rect

    @staticmethod
    def _point_in_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = rect
        return x <= point[0] <= x + w and y <= point[1] <= y + h

    def _apply_step(self, index: int, direction: float) -> None:
        item = self.items[index]
        value = item.get_value() + item.step * direction
        value = max(item.min_value, min(item.max_value, value))
        if item.integer:
            value = int(round(value))
        item.set_value(value)

    def _draw_item(
        self,
        frame,
        item: MenuItem,
        rect: Tuple[int, int, int, int],
        active: bool,
        index: int,
    ) -> None:
        if item.kind == "button":
            self._draw_button_row(frame, item, rect, active, index)
            return
        x, y, w, h = rect
        row_radius = max(8, int(min(h * 0.45, 12 * self._scale)))
        accent = IOS_BLUE
        track = IOS_TRACK
        text_color = IOS_TEXT
        slider_left, slider_right, slider_center_y = self._slider_bounds(rect)
        minus_rect, plus_rect = self._button_rects(rect)
        track_thickness = max(3, int(4 * self._scale))
        if active:
            overlay = frame.copy()
            draw_rounded_rect(overlay, (x, y, w, h), row_radius, IOS_GLASS_STRONG, -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        draw_text(
            frame,
            item.label,
            (x + 6, slider_center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, 0.5 * self._scale),
            text_color,
            max(1, int(1 * self._scale)),
            shadow=False,
        )
        if slider_right > slider_left:
            cv2.line(
                frame,
                (slider_left, slider_center_y),
                (slider_right, slider_center_y),
                track,
                track_thickness,
            )
        value = item.get_value()
        ratio = (value - item.min_value) / (item.max_value - item.min_value or 1.0)
        ratio = max(0.0, min(1.0, ratio))
        if slider_right > slider_left:
            knob_x = int(slider_left + ratio * (slider_right - slider_left))
            knob_radius = max(8, int(8 * self._scale))
            cv2.line(
                frame,
                (slider_left, slider_center_y),
                (knob_x, slider_center_y),
                accent,
                track_thickness,
            )
            cv2.circle(
                frame,
                (knob_x, slider_center_y + 1),
                knob_radius + 1,
                IOS_SHADOW,
                -1,
            )
            cv2.circle(frame, (knob_x, slider_center_y), knob_radius, IOS_KNOB, -1)
            cv2.circle(frame, (knob_x, slider_center_y), knob_radius, IOS_BORDER, 1)
        value_text = item.fmt.format(value)
        value_x = max(slider_left, minus_rect[0] - int(60 * self._scale))
        draw_text(
            frame,
            value_text,
            (value_x, slider_center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, 0.45 * self._scale),
            IOS_TEXT_MUTED,
            max(1, int(1 * self._scale)),
            shadow=False,
        )

        self._draw_button(frame, minus_rect, "-")
        self._draw_button(frame, plus_rect, "+")
        if index < len(self.items) - 1:
            line_y = y + h + int(4 * self._scale)
            cv2.line(frame, (x + 6, line_y), (x + w - 6, line_y), IOS_SEPARATOR, 1)

    def _draw_button_row(
        self,
        frame,
        item: MenuItem,
        rect: Tuple[int, int, int, int],
        active: bool,
        index: int,
    ) -> None:
        x, y, w, h = rect
        is_on = item.state() if item.state is not None else False
        row_radius = max(8, int(min(h * 0.45, 12 * self._scale)))
        if active:
            overlay = frame.copy()
            draw_rounded_rect(overlay, (x, y, w, h), row_radius, IOS_GLASS_STRONG, -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        draw_text(
            frame,
            item.label,
            (x + 6, y + int(h * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, 0.5 * self._scale),
            IOS_TEXT,
            max(1, int(1 * self._scale)),
            shadow=False,
        )

        switch_w = int(max(44, 46 * self._scale))
        switch_h = int(max(24, 26 * self._scale))
        switch_x = x + w - switch_w
        switch_y = y + (h - switch_h) // 2
        switch_radius = max(1, switch_h // 2)
        track_color = IOS_BLUE if is_on else IOS_TRACK
        draw_rounded_rect(
            frame,
            (switch_x, switch_y, switch_w, switch_h),
            switch_radius,
            track_color,
            -1,
        )
        draw_rounded_rect(
            frame,
            (switch_x, switch_y, switch_w, switch_h),
            switch_radius,
            IOS_BORDER,
            1,
        )
        knob_radius = max(4, switch_radius - 2)
        knob_x = switch_x + switch_w - switch_radius if is_on else switch_x + switch_radius
        cv2.circle(frame, (knob_x, switch_y + switch_radius + 1), knob_radius, IOS_SHADOW, -1)
        cv2.circle(frame, (knob_x, switch_y + switch_radius), knob_radius, IOS_KNOB, -1)
        cv2.circle(frame, (knob_x, switch_y + switch_radius), knob_radius, IOS_BORDER, 1)
        if index < len(self.items) - 1:
            line_y = y + h + int(4 * self._scale)
            cv2.line(frame, (x + 6, line_y), (x + w - 6, line_y), IOS_SEPARATOR, 1)

    def _draw_button(self, frame, rect: Tuple[int, int, int, int], label: str) -> None:
        x, y, w, h = rect
        radius = max(6, int(min(w, h) * 0.35))
        draw_rounded_rect(frame, (x, y, w, h), radius, IOS_GLASS_STRONG, -1)
        draw_rounded_rect(frame, (x, y, w, h), radius, IOS_BORDER, 1)
        font_scale = max(0.6, 0.6 * self._scale)
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
            max(1, int(1 * self._scale)),
            shadow=False,
        )
