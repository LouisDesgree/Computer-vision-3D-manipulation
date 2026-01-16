from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import math
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .hand_tracking import HandTracker, HAND_CONNECTIONS
from .palette import IOS_BLUE, IOS_BLUE_SOFT, IOS_KNOB


@dataclass
class HandState:
    id: int
    landmarks_2d: List[Tuple[int, int]]
    landmarks_3d: List[Tuple[float, float, float]]
    hull: Optional[np.ndarray]
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    pinch: bool
    pinch_strength: float
    grip: bool
    grip_strength: float
    open_palm: bool
    open_strength: float
    press: bool
    press_strength: float


class HandInput:
    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[Path] = None,
        mode: str = "hybrid",
        glove_lower: Tuple[int, int, int] = (90, 60, 50),
        glove_upper: Tuple[int, int, int] = (130, 255, 255),
        glove_min_area: float = 800.0,
        glove_kernel_size: int = 3,
        glove_open: int = 0,
        glove_close: int = 1,
        glove_dilate: int = 1,
        glove_contour_epsilon: float = 0.008,
        pinch_ratio: float = 0.45,
        grip_ratio: float = 1.1,
        open_ratio: float = 1.6,
        press_depth: float = 0.05,
    ) -> None:
        self._tracker = HandTracker(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=model_path,
        )
        self._mode = mode
        self._max_hands = max_hands
        self._glove_lower = np.array(glove_lower, dtype=np.uint8)
        self._glove_upper = np.array(glove_upper, dtype=np.uint8)
        self._glove_min_area = glove_min_area
        self._glove_kernel_size = max(1, glove_kernel_size)
        self._glove_open = max(0, glove_open)
        self._glove_close = max(0, glove_close)
        self._glove_dilate = max(0, glove_dilate)
        self._glove_contour_epsilon = max(0.0, glove_contour_epsilon)
        self._pinch_ratio = max(0.05, pinch_ratio)
        self._grip_ratio = max(0.6, grip_ratio)
        self._open_ratio = max(1.0, open_ratio)
        self._press_depth = max(0.005, press_depth)
        self._last_centers: List[Tuple[float, float]] = []
        self._last_ids: List[int] = []
        self._next_id: int = 1
        self._last_time: float = 0.0
        self._last_results = None
        self._last_hands: List[HandState] = []
        self._last_glove_hulls: List[np.ndarray] = []
        self._last_glove_centers: List[Tuple[int, int]] = []
        self._last_mask: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def update(self, frame) -> List[HandState]:
        if self._mode in ("mediapipe", "hybrid"):
            results = self._tracker.process(frame)
            if results:
                hands = self._build_from_results(results)
                with self._lock:
                    self._last_results = results
                    self._last_glove_hulls = []
                    self._last_glove_centers = []
                    self._last_mask = None
                    self._last_hands = self._assign_velocities(hands)
                    return list(self._last_hands)
            if self._mode == "mediapipe":
                with self._lock:
                    self._clear_tracking()
                return []

        hands, glove_hulls, glove_centers, mask = self._detect_gloves(frame)
        with self._lock:
            self._last_results = None
            self._last_glove_hulls = glove_hulls
            self._last_glove_centers = glove_centers
            self._last_mask = mask
            self._last_hands = self._assign_velocities(hands)
            return list(self._last_hands)

    def contact_distances(
        self,
        hands: List[HandState],
        point: Tuple[float, float],
    ) -> List[Optional[float]]:
        distances: List[Optional[float]] = []
        for hand in hands:
            distance = self._distance_to_hand(hand, point)
            distances.append(distance)
        return distances

    def contact_vectors(
        self,
        hands: List[HandState],
        point: Tuple[float, float],
    ) -> List[Optional[Tuple[float, Tuple[float, float]]]]:
        infos: List[Optional[Tuple[float, Tuple[float, float]]]] = []
        for hand in hands:
            info = self._contact_vector(hand, point)
            infos.append(info)
        return infos

    def draw(self, frame) -> None:
        with self._lock:
            hands = list(self._last_hands)
            glove_hulls = list(self._last_glove_hulls)
            glove_centers = list(self._last_glove_centers)

        if any(hand.landmarks_3d for hand in hands):
            for hand in hands:
                self._draw_stylized(frame, hand.landmarks_2d)
            return

        for outline in glove_hulls:
            cv2.polylines(frame, [outline], True, IOS_BLUE, 2)
        for center in glove_centers:
            cv2.circle(frame, center, 6, IOS_BLUE_SOFT, -1)

    def last_mask(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._last_mask

    def last_hulls(self) -> List[np.ndarray]:
        with self._lock:
            hands = list(self._last_hands)
            glove_hulls = list(self._last_glove_hulls)
        hulls: List[np.ndarray] = []
        for hand in hands:
            if hand.hull is not None:
                hulls.append(hand.hull)
        if not hulls:
            hulls = glove_hulls
        return hulls

    def stylized_mask(self, frame_shape) -> Optional[np.ndarray]:
        with self._lock:
            hands = list(self._last_hands)
            glove_hulls = list(self._last_glove_hulls)
        if not hands:
            return None
        height, width = frame_shape[:2]
        base, line_outer, _, dot_outer, _, _ = self._stylized_sizes(height, width)
        mask = np.zeros((height, width), dtype=np.uint8)
        if any(hand.landmarks_3d for hand in hands):
            for hand in hands:
                points = hand.landmarks_2d
                for start, end in HAND_CONNECTIONS:
                    cv2.line(mask, points[start], points[end], 255, line_outer)
                for point in points:
                    cv2.circle(mask, point, dot_outer, 255, -1)
            return mask
        for outline in glove_hulls:
            cv2.polylines(mask, [outline], True, 255, max(1, base))
        return mask

    def close(self) -> None:
        self._tracker.close()

    def _build_from_results(self, results) -> List[HandState]:
        hands: List[HandState] = []
        for hand in results[: self._max_hands]:
            points = hand.landmarks_2d
            if not points:
                continue
            pinch, pinch_strength = self._compute_pinch(points)
            grip, grip_strength = self._compute_grip(points)
            open_palm, open_strength = self._compute_open_palm(points)
            press, press_strength = self._compute_press(hand.landmarks_3d)
            center = (
                sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points),
            )
            hull = None
            if len(points) >= 3:
                hull = cv2.convexHull(np.array(points, dtype=np.int32).reshape((-1, 1, 2)))
            hands.append(
                HandState(
                    id=0,
                    landmarks_2d=points,
                    landmarks_3d=hand.landmarks_3d,
                    hull=hull,
                    center=center,
                    velocity=(0.0, 0.0),
                    pinch=pinch,
                    pinch_strength=pinch_strength,
                    grip=grip,
                    grip_strength=grip_strength,
                    open_palm=open_palm,
                    open_strength=open_strength,
                    press=press,
                    press_strength=press_strength,
                )
            )
        return hands

    def _detect_gloves(
        self, frame
    ) -> Tuple[List[HandState], List[np.ndarray], List[Tuple[int, int]], np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._glove_lower, self._glove_upper)
        kernel = np.ones((self._glove_kernel_size, self._glove_kernel_size), dtype=np.uint8)
        if self._glove_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self._glove_open)
        if self._glove_close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self._glove_close)
        if self._glove_dilate:
            mask = cv2.dilate(mask, kernel, iterations=self._glove_dilate)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = max(self._glove_min_area, frame_area * 0.0006)
        candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
        candidates.sort(key=cv2.contourArea, reverse=True)
        candidates = candidates[: self._max_hands]

        hands: List[HandState] = []
        glove_hulls: List[np.ndarray] = []
        glove_centers: List[Tuple[int, int]] = []
        for contour in candidates:
            outline = contour
            if self._glove_contour_epsilon > 0.0:
                epsilon = self._glove_contour_epsilon * cv2.arcLength(contour, True)
                outline = cv2.approxPolyDP(contour, epsilon, True)
            moments = cv2.moments(outline)
            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                pts = outline.reshape((-1, 2))
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
            center = (cx, cy)
            landmarks = outline.reshape((-1, 2)).astype(int).tolist()
            hands.append(
                HandState(
                    id=0,
                    landmarks_2d=landmarks,
                    landmarks_3d=[],
                    hull=outline,
                    center=center,
                    velocity=(0.0, 0.0),
                    pinch=False,
                    pinch_strength=0.0,
                    grip=False,
                    grip_strength=0.0,
                    open_palm=False,
                    open_strength=0.0,
                    press=False,
                    press_strength=0.0,
                )
            )
            glove_hulls.append(outline)
            glove_centers.append((int(cx), int(cy)))

        return hands, glove_hulls, glove_centers, mask

    def _assign_velocities(self, hands: List[HandState]) -> List[HandState]:
        if not hands:
            self._clear_tracking()
            return []

        now = time.time()
        velocities = [(0.0, 0.0) for _ in hands]
        assigned_ids: List[Optional[int]] = [None for _ in hands]
        if self._last_centers and self._last_ids and self._last_time > 0.0:
            dt = max(now - self._last_time, 1e-6)
            prev = list(zip(self._last_ids, self._last_centers))
            for idx, hand in enumerate(hands):
                if prev:
                    nearest = min(
                        range(len(prev)),
                        key=lambda i: (
                            (hand.center[0] - prev[i][1][0]) ** 2
                            + (hand.center[1] - prev[i][1][1]) ** 2
                        ),
                    )
                    prev_id, prev_center = prev.pop(nearest)
                    assigned_ids[idx] = prev_id
                    velocities[idx] = (
                        (hand.center[0] - prev_center[0]) / dt,
                        (hand.center[1] - prev_center[1]) / dt,
                    )

        for idx, hand in enumerate(hands):
            if assigned_ids[idx] is None:
                assigned_ids[idx] = self._next_id
                self._next_id += 1

        self._last_centers = [hand.center for hand in hands]
        self._last_ids = [int(hand_id) for hand_id in assigned_ids]
        self._last_time = now
        updated: List[HandState] = []
        for hand, velocity, hand_id in zip(hands, velocities, assigned_ids):
            updated.append(
                HandState(
                    id=int(hand_id) if hand_id is not None else 0,
                    landmarks_2d=hand.landmarks_2d,
                    landmarks_3d=hand.landmarks_3d,
                    hull=hand.hull,
                    center=hand.center,
                    velocity=velocity,
                    pinch=hand.pinch,
                    pinch_strength=hand.pinch_strength,
                    grip=hand.grip,
                    grip_strength=hand.grip_strength,
                    open_palm=hand.open_palm,
                    open_strength=hand.open_strength,
                    press=hand.press,
                    press_strength=hand.press_strength,
                )
            )
        return updated

    def _clear_tracking(self) -> None:
        self._last_centers = []
        self._last_ids = []
        self._last_time = 0.0
        self._last_results = None
        self._last_hands = []
        self._last_glove_hulls = []
        self._last_glove_centers = []
        self._last_mask = None

    def _compute_pinch(self, points: List[Tuple[int, int]]) -> Tuple[bool, float]:
        if len(points) <= 9:
            return False, 0.0
        thumb = points[4]
        index_tip = points[8]
        wrist = points[0]
        mid_mcp = points[9]
        palm_size = math.hypot(mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1])
        if palm_size < 1.0:
            return False, 0.0
        pinch_dist = math.hypot(index_tip[0] - thumb[0], index_tip[1] - thumb[1])
        ratio = pinch_dist / palm_size
        pinch_strength = max(0.0, 1.0 - ratio / self._pinch_ratio)
        return ratio < self._pinch_ratio, pinch_strength

    def _compute_grip(self, points: List[Tuple[int, int]]) -> Tuple[bool, float]:
        if len(points) <= 20:
            return False, 0.0
        ratio = self._palm_ratio(points)
        if ratio is None:
            return False, 0.0
        grip_strength = max(0.0, min(1.0, 1.0 - ratio / self._grip_ratio))
        return ratio < self._grip_ratio, grip_strength

    def _compute_open_palm(self, points: List[Tuple[int, int]]) -> Tuple[bool, float]:
        if len(points) <= 20:
            return False, 0.0
        ratio = self._palm_ratio(points)
        if ratio is None:
            return False, 0.0
        open_strength = max(0.0, min(1.0, (ratio - self._open_ratio) / self._open_ratio))
        return ratio > self._open_ratio, open_strength

    def _compute_press(
        self, points_3d: List[Tuple[float, float, float]]
    ) -> Tuple[bool, float]:
        if len(points_3d) <= 8:
            return False, 0.0
        palm_indices = (0, 5, 9, 13, 17)
        palm_z = sum(points_3d[idx][2] for idx in palm_indices) / len(palm_indices)
        tip_z = points_3d[8][2]
        depth = palm_z - tip_z
        press_strength = max(0.0, min(1.0, (depth - self._press_depth) / self._press_depth))
        return depth > self._press_depth, press_strength

    @staticmethod
    def _palm_ratio(points: List[Tuple[int, int]]) -> Optional[float]:
        palm_indices = (0, 5, 9, 13, 17)
        mcp_indices = (5, 9, 13, 17)
        tip_indices = (4, 8, 12, 16, 20)
        palm_center = (
            sum(points[idx][0] for idx in palm_indices) / len(palm_indices),
            sum(points[idx][1] for idx in palm_indices) / len(palm_indices),
        )
        mcp_dist = sum(
            math.hypot(points[idx][0] - palm_center[0], points[idx][1] - palm_center[1])
            for idx in mcp_indices
        ) / len(mcp_indices)
        tip_dist = sum(
            math.hypot(points[idx][0] - palm_center[0], points[idx][1] - palm_center[1])
            for idx in tip_indices
        ) / len(tip_indices)
        if mcp_dist < 1.0:
            return None
        return tip_dist / mcp_dist

    def set_pinch_ratio(self, value: float) -> None:
        self._pinch_ratio = max(0.05, float(value))

    def get_pinch_ratio(self) -> float:
        return float(self._pinch_ratio)

    def set_grip_ratio(self, value: float) -> None:
        self._grip_ratio = max(0.6, float(value))

    def get_grip_ratio(self) -> float:
        return float(self._grip_ratio)

    def _draw_stylized(self, frame, landmarks: List[Tuple[int, int]]) -> None:
        if not landmarks:
            return
        height, width = frame.shape[:2]
        base, line_outer, line_inner, dot_outer, dot_inner, dot_highlight = (
            self._stylized_sizes(height, width)
        )

        outer_color = (18, 18, 24)
        inner_color = IOS_BLUE
        highlight_color = IOS_KNOB

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, landmarks[start], landmarks[end], outer_color, line_outer)
            cv2.line(frame, landmarks[start], landmarks[end], inner_color, line_inner)

        for point in landmarks:
            cv2.circle(frame, point, dot_outer, outer_color, -1)
            cv2.circle(frame, point, dot_inner, inner_color, -1)
            cv2.circle(frame, point, dot_highlight, highlight_color, -1)

    def _stylized_sizes(
        self, height: int, width: int
    ) -> Tuple[int, int, int, int, int, int]:
        base = max(2, int(min(height, width) * 0.004))
        line_outer = base * 2
        line_inner = max(1, base)
        dot_outer = base * 3
        dot_inner = base * 2
        dot_highlight = max(1, base // 2)
        return base, line_outer, line_inner, dot_outer, dot_inner, dot_highlight

    def _distance_to_hand(
        self,
        hand: HandState,
        point: Tuple[float, float],
    ) -> Optional[float]:
        if hand.landmarks_3d and len(hand.landmarks_2d) >= 21:
            min_dist = None
            for start, end in HAND_CONNECTIONS:
                if start >= len(hand.landmarks_2d) or end >= len(hand.landmarks_2d):
                    continue
                dist = self._point_segment_distance(
                    point,
                    hand.landmarks_2d[start],
                    hand.landmarks_2d[end],
                )
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            for landmark in hand.landmarks_2d:
                dist = math.hypot(point[0] - landmark[0], point[1] - landmark[1])
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            return min_dist
        if hand.hull is None:
            return None
        return abs(cv2.pointPolygonTest(hand.hull, point, True))

    def _contact_vector(
        self,
        hand: HandState,
        point: Tuple[float, float],
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        if hand.landmarks_3d and len(hand.landmarks_2d) >= 21:
            min_dist = None
            closest = None
            for start, end in HAND_CONNECTIONS:
                if start >= len(hand.landmarks_2d) or end >= len(hand.landmarks_2d):
                    continue
                dist, proj = self._point_segment_closest(
                    point,
                    hand.landmarks_2d[start],
                    hand.landmarks_2d[end],
                )
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    closest = proj
            for landmark in hand.landmarks_2d:
                dist = math.hypot(point[0] - landmark[0], point[1] - landmark[1])
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    closest = (float(landmark[0]), float(landmark[1]))
            if min_dist is None or closest is None:
                return None
            normal_x = point[0] - closest[0]
            normal_y = point[1] - closest[1]
            norm = math.hypot(normal_x, normal_y)
            if norm > 1e-6:
                normal_x /= norm
                normal_y /= norm
            else:
                normal_x = point[0] - hand.center[0]
                normal_y = point[1] - hand.center[1]
                norm = math.hypot(normal_x, normal_y)
                if norm > 1e-6:
                    normal_x /= norm
                    normal_y /= norm
                else:
                    normal_x, normal_y = 1.0, 0.0
            return min_dist, (normal_x, normal_y)
        if hand.hull is None:
            return None
        pts = hand.hull.reshape((-1, 2))
        if len(pts) < 2:
            return None
        min_dist = None
        closest = None
        for idx in range(len(pts)):
            a = tuple(int(v) for v in pts[idx])
            b = tuple(int(v) for v in pts[(idx + 1) % len(pts)])
            dist, proj = self._point_segment_closest(point, a, b)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                closest = proj
        if min_dist is None or closest is None:
            return None
        normal_x = point[0] - closest[0]
        normal_y = point[1] - closest[1]
        norm = math.hypot(normal_x, normal_y)
        if norm > 1e-6:
            normal_x /= norm
            normal_y /= norm
        else:
            normal_x = point[0] - hand.center[0]
            normal_y = point[1] - hand.center[1]
            norm = math.hypot(normal_x, normal_y)
            if norm > 1e-6:
                normal_x /= norm
                normal_y /= norm
            else:
                normal_x, normal_y = 1.0, 0.0
        signed = cv2.pointPolygonTest(hand.hull, point, True)
        if signed > 0:
            normal_x = -normal_x
            normal_y = -normal_y
        return min_dist, (normal_x, normal_y)

    @staticmethod
    def _point_segment_distance(
        point: Tuple[float, float],
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> float:
        px, py = point
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    @staticmethod
    def _point_segment_closest(
        point: Tuple[float, float],
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> Tuple[float, Tuple[float, float]]:
        px, py = point
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay), (float(ax), float(ay))
        t = ((px - ax) * dx + (py - ay) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y), (float(proj_x), float(proj_y))
