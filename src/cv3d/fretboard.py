from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .palette import IOS_BLUE, IOS_BLUE_SOFT, IOS_BORDER, IOS_TEXT
from .ui import draw_text


@dataclass
class FretboardConfig:
    scale: float = 0.6
    canny_low: int = 50
    canny_high: int = 140
    min_area_ratio: float = 0.03
    min_aspect: float = 3.0
    angle_tol: float = 12.0
    min_line_length: int = 40
    max_line_gap: int = 10
    string_count: int = 6
    string_cluster_ratio: float = 0.04
    fret_cluster_ratio: float = 0.025
    smooth_alpha: float = 0.5
    hold_seconds: float = 0.7
    line_angle_bin: float = 6.0
    line_trim_ratio: float = 0.08
    line_min_count: int = 6
    line_candidates: int = 3
    min_fret_lines: int = 3
    min_string_lines: int = 2
    fret_pad_ratio: float = 0.12
    string_pad_ratio: float = 0.18
    mask_use_color: bool = False
    mask_color_lower: Tuple[int, int, int] = (5, 30, 40)
    mask_color_upper: Tuple[int, int, int] = (30, 255, 255)
    mask_color_open: int = 0
    mask_color_close: int = 1
    mask_color_dilate: int = 1
    mask_use_depth: bool = False
    mask_depth_blur: int = 5
    mask_depth_threshold: int = 18
    mask_depth_dilate: int = 1
    mask_exclude_hands: bool = False
    mask_hand_dilate: int = 18
    hand_refine_margin_ratio: float = 0.45
    hand_refine_min_points: int = 2
    hand_refine_shift_len_ratio: float = 0.25
    hand_refine_shift_width_ratio: float = 0.4


@dataclass
class FretboardResult:
    polygon: np.ndarray
    origin: Tuple[float, float]
    length_dir: Tuple[float, float]
    width_dir: Tuple[float, float]
    length: float
    width: float
    string_positions: List[float]
    fret_positions: List[float]
    fret_positions_from_nut: List[float]
    origin_is_nut: bool


@dataclass
class FingerPlacement:
    name: str
    point: Tuple[int, int]
    string_index: Optional[int]
    fret_index: Optional[int]


@dataclass
class LineInfo:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    midpoint: Tuple[float, float]
    angle: float
    length: float


FINGER_TIPS = {
    "T": 4,
    "I": 8,
    "M": 12,
    "R": 16,
    "P": 20,
}

PALM_INDICES = (0, 5, 9, 13, 17)


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _compute_axes(polygon: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float, float]:
    ordered = _order_points(polygon)
    tl, tr, br, bl = ordered
    edge1 = tr - tl
    edge2 = bl - tl
    len1 = float(np.linalg.norm(edge1))
    len2 = float(np.linalg.norm(edge2))
    if len1 >= len2:
        length_dir = edge1 / max(len1, 1e-6)
        width_dir = edge2 / max(len2, 1e-6)
    else:
        length_dir = edge2 / max(len2, 1e-6)
        width_dir = edge1 / max(len1, 1e-6)

    proj_len = [float(np.dot(pt, length_dir)) for pt in ordered]
    proj_w = [float(np.dot(pt, width_dir)) for pt in ordered]
    min_len, max_len = min(proj_len), max(proj_len)
    min_w, max_w = min(proj_w), max(proj_w)
    origin = (
        float(length_dir[0] * min_len + width_dir[0] * min_w),
        float(length_dir[1] * min_len + width_dir[1] * min_w),
    )
    length = max_len - min_len
    width = max_w - min_w
    return origin, (float(length_dir[0]), float(length_dir[1])), (float(width_dir[0]), float(width_dir[1])), float(length), float(width)


def _rect_from_axes(
    origin: Tuple[float, float],
    length_dir: Tuple[float, float],
    width_dir: Tuple[float, float],
    length: float,
    width: float,
) -> np.ndarray:
    tl = np.array(origin, dtype=np.float32)
    tr = tl + np.array(length_dir, dtype=np.float32) * float(length)
    bl = tl + np.array(width_dir, dtype=np.float32) * float(width)
    br = tr + np.array(width_dir, dtype=np.float32) * float(width)
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _cluster_positions(values: List[float], tol: float) -> List[float]:
    if not values:
        return []
    values = sorted(values)
    clusters: List[List[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - clusters[-1][-1]) <= tol:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _normalize_angle(angle: float) -> float:
    angle = angle % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def _angle_diff(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b) % 180.0
    return min(diff, 180.0 - diff)


def _dominant_angles(
    lines: List[LineInfo], bin_size: float, top_k: int
) -> List[float]:
    if not lines:
        return []
    bins = max(1, int(180.0 / max(bin_size, 1.0)))
    weights = [0.0 for _ in range(bins)]
    for line in lines:
        idx = int(line.angle / bin_size) % bins
        weights[idx] += line.length
    ranked = sorted(range(bins), key=lambda i: weights[i], reverse=True)
    angles: List[float] = []
    for idx in ranked:
        if weights[idx] <= 0:
            continue
        angle = (idx + 0.5) * bin_size
        if any(_angle_diff(angle, other) <= bin_size for other in angles):
            continue
        angles.append(angle)
        if len(angles) >= top_k:
            break
    return angles


def _average_direction(lines: List[LineInfo]) -> Optional[Tuple[float, float]]:
    if not lines:
        return None
    ref = None
    acc_x = 0.0
    acc_y = 0.0
    for line in lines:
        dx = line.p2[0] - line.p1[0]
        dy = line.p2[1] - line.p1[1]
        length = line.length
        if length <= 1e-6:
            continue
        vx = dx / length
        vy = dy / length
        if ref is None:
            ref = (vx, vy)
        elif vx * ref[0] + vy * ref[1] < 0:
            vx = -vx
            vy = -vy
        acc_x += vx
        acc_y += vy
    norm = math.hypot(acc_x, acc_y)
    if norm <= 1e-6:
        return None
    return (acc_x / norm, acc_y / norm)


def _trim_bounds(values: List[float], trim_ratio: float) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    values = sorted(values)
    if len(values) < 4:
        return (values[0], values[-1])
    trim = int(len(values) * trim_ratio)
    trim = min(trim, max(0, len(values) // 2 - 1))
    return (values[trim], values[-trim - 1])


def _project_values(
    points: List[Tuple[float, float]], direction: Tuple[float, float]
) -> List[float]:
    if not points:
        return []
    dx, dy = direction
    return [pt[0] * dx + pt[1] * dy for pt in points]


def _collect_line_info(lines, scale: float) -> List[LineInfo]:
    results: List[LineInfo] = []
    if lines is None:
        return results
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(val) / scale for val in line]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-3:
            continue
        angle = _normalize_angle(math.degrees(math.atan2(dy, dx)))
        midpoint = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        results.append(
            LineInfo(
                p1=(x1, y1),
                p2=(x2, y2),
                midpoint=midpoint,
                angle=angle,
                length=length,
            )
        )
    return results


class FretboardDetector:
    def __init__(self, config: Optional[FretboardConfig] = None) -> None:
        self.config = config or FretboardConfig()

    @staticmethod
    def _odd_kernel(value: int, minimum: int = 1) -> int:
        size = max(minimum, int(value))
        if size % 2 == 0:
            size += 1
        return size

    def _build_color_mask(self, hsv) -> np.ndarray:
        lower = np.array(self.config.mask_color_lower, dtype=np.uint8)
        upper = np.array(self.config.mask_color_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), dtype=np.uint8)
        open_iter = max(0, int(self.config.mask_color_open))
        close_iter = max(0, int(self.config.mask_color_close))
        dilate_iter = max(0, int(self.config.mask_color_dilate))
        if open_iter > 0:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=open_iter
            )
        if close_iter > 0:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter
            )
        if dilate_iter > 0:
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        return mask

    def _build_depth_mask(self, gray) -> np.ndarray:
        blur_size = self._odd_kernel(self.config.mask_depth_blur, minimum=3)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        lap = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
        abs_lap = cv2.convertScaleAbs(lap)
        thresh = max(1, int(self.config.mask_depth_threshold))
        _, mask = cv2.threshold(abs_lap, thresh, 255, cv2.THRESH_BINARY)
        dilate_iter = max(0, int(self.config.mask_depth_dilate))
        if dilate_iter > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        return mask

    def _combine_masks(
        self,
        base_mask: Optional[np.ndarray],
        add_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if add_mask is None:
            return base_mask
        if base_mask is None:
            return add_mask
        return cv2.bitwise_and(base_mask, add_mask)

    def build_mask(
        self,
        frame,
        include_mask: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        height, width = frame.shape[:2]
        scale = max(0.3, min(1.0, float(self.config.scale)))
        small = cv2.resize(frame, (int(width * scale), int(height * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        mask = None
        if include_mask is not None:
            include_small = cv2.resize(
                include_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            mask = self._combine_masks(mask, include_small)
        if self.config.mask_use_color:
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            color_mask = self._build_color_mask(hsv)
            mask = self._combine_masks(mask, color_mask)
        if self.config.mask_use_depth:
            depth_mask = self._build_depth_mask(gray)
            mask = self._combine_masks(mask, depth_mask)
        if exclude_mask is not None:
            exclude_small = cv2.resize(
                exclude_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            if mask is None:
                mask = np.ones_like(exclude_small)
            mask[exclude_small > 0] = 0

        if mask is None:
            return None
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def _detect_from_lines(self, edges, scale: float, frame_shape) -> Optional[FretboardResult]:
        lines = cv2.HoughLinesP(
            edges,
            1,
            math.pi / 180.0,
            threshold=70,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )
        line_info = _collect_line_info(lines, scale)
        if len(line_info) < self.config.line_min_count:
            return None
        candidate_angles = _dominant_angles(
            line_info, self.config.line_angle_bin, self.config.line_candidates
        )
        if not candidate_angles:
            return None

        best_result = None
        best_score = 0.0
        for angle in candidate_angles:
            result, score = self._try_line_angle(line_info, angle, frame_shape)
            if result is None:
                continue
            if score > best_score:
                best_result = result
                best_score = score
        return best_result

    def _try_line_angle(
        self, line_info: List[LineInfo], angle: float, frame_shape
    ) -> Tuple[Optional[FretboardResult], float]:
        angle_tol = self.config.angle_tol
        string_lines = [
            line for line in line_info if _angle_diff(line.angle, angle) <= angle_tol
        ]
        fret_lines = [
            line
            for line in line_info
            if _angle_diff(line.angle, angle + 90.0) <= angle_tol
        ]
        if len(string_lines) < 2:
            return None, 0.0

        length_dir = _average_direction(string_lines)
        if length_dir is None:
            return None, 0.0
        width_dir = (-length_dir[1], length_dir[0])

        points: List[Tuple[float, float]] = []
        for line in string_lines + fret_lines:
            points.append(line.p1)
            points.append(line.p2)
        if len(points) < 4:
            return None, 0.0

        proj_len_all = _project_values(points, length_dir)
        proj_w_all = _project_values(points, width_dir)
        min_len_all, max_len_all = _trim_bounds(proj_len_all, self.config.line_trim_ratio)
        min_w_all, max_w_all = _trim_bounds(proj_w_all, self.config.line_trim_ratio)

        fret_mids = [line.midpoint for line in fret_lines]
        string_mids = [line.midpoint for line in string_lines]
        proj_len_frets = _project_values(fret_mids, length_dir)
        proj_w_strings = _project_values(string_mids, width_dir)

        if len(proj_len_frets) >= self.config.min_fret_lines:
            min_len, max_len = _trim_bounds(proj_len_frets, self.config.line_trim_ratio)
        else:
            min_len, max_len = min_len_all, max_len_all

        if len(proj_w_strings) >= self.config.min_string_lines:
            min_w, max_w = _trim_bounds(proj_w_strings, self.config.line_trim_ratio)
        else:
            min_w, max_w = min_w_all, max_w_all

        length = max_len - min_len
        board_width = max_w - min_w
        if length <= 1.0 or board_width <= 1.0:
            return None, 0.0
        ratio = max(length, board_width) / max(1.0, min(length, board_width))
        if ratio < self.config.min_aspect * 0.7:
            return None, 0.0

        frame_area = frame_shape[0] * frame_shape[1]
        if length * board_width < frame_area * self.config.min_area_ratio * 0.5:
            return None, 0.0

        pad_len = max(4.0, length * self.config.fret_pad_ratio)
        pad_w = max(6.0, board_width * self.config.string_pad_ratio)
        min_len -= pad_len
        max_len += pad_len
        min_w -= pad_w
        max_w += pad_w
        length = max_len - min_len
        board_width = max_w - min_w

        origin = (
            float(length_dir[0] * min_len + width_dir[0] * min_w),
            float(length_dir[1] * min_len + width_dir[1] * min_w),
        )
        polygon = _rect_from_axes(origin, length_dir, width_dir, length, board_width)

        string_positions: List[float] = []
        for line in string_lines:
            rel = (line.midpoint[0] - origin[0], line.midpoint[1] - origin[1])
            pos = rel[0] * width_dir[0] + rel[1] * width_dir[1]
            if 0 <= pos <= board_width:
                string_positions.append(pos)

        fret_positions: List[float] = []
        for line in fret_lines:
            rel = (line.midpoint[0] - origin[0], line.midpoint[1] - origin[1])
            pos = rel[0] * length_dir[0] + rel[1] * length_dir[1]
            if 0 <= pos <= length:
                fret_positions.append(pos)

        string_tol = max(4.0, board_width * self.config.string_cluster_ratio)
        fret_tol = max(6.0, length * self.config.fret_cluster_ratio)
        string_positions = _cluster_positions(string_positions, string_tol)
        fret_positions = _cluster_positions(fret_positions, fret_tol)
        string_positions.sort()
        fret_positions.sort()

        if len(string_positions) < max(3, self.config.string_count // 2):
            step = board_width / max(1, self.config.string_count - 1)
            string_positions = [idx * step for idx in range(self.config.string_count)]

        origin_is_nut = True
        fret_positions_from_nut = list(fret_positions)
        if len(fret_positions) >= 2:
            first_gap = fret_positions[1] - fret_positions[0]
            last_gap = fret_positions[-1] - fret_positions[-2]
            origin_is_nut = first_gap > last_gap
            if not origin_is_nut:
                fret_positions_from_nut = [length - pos for pos in reversed(fret_positions)]

        score = sum(line.length for line in string_lines) + sum(
            line.length for line in fret_lines
        ) * 0.7
        return (
            FretboardResult(
                polygon=polygon,
                origin=origin,
                length_dir=length_dir,
                width_dir=width_dir,
                length=length,
                width=board_width,
                string_positions=string_positions,
                fret_positions=fret_positions,
                fret_positions_from_nut=fret_positions_from_nut,
                origin_is_nut=origin_is_nut,
            ),
            score,
        )

    def detect(
        self,
        frame,
        include_mask: Optional[np.ndarray] = None,
        exclude_mask: Optional[np.ndarray] = None,
    ) -> Optional[FretboardResult]:
        height, width = frame.shape[:2]
        scale = max(0.3, min(1.0, float(self.config.scale)))
        small = cv2.resize(frame, (int(width * scale), int(height * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.config.canny_low, self.config.canny_high)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

        mask = None
        if include_mask is not None:
            include_small = cv2.resize(
                include_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            mask = self._combine_masks(mask, include_small)
        if self.config.mask_use_color:
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            color_mask = self._build_color_mask(hsv)
            mask = self._combine_masks(mask, color_mask)
        if self.config.mask_use_depth:
            depth_mask = self._build_depth_mask(gray)
            mask = self._combine_masks(mask, depth_mask)
        if exclude_mask is not None:
            exclude_small = cv2.resize(
                exclude_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            if mask is None:
                mask = np.ones_like(edges)
            mask[exclude_small > 0] = 0

        if mask is not None:
            edges = cv2.bitwise_and(edges, edges, mask=mask)

        line_result = self._detect_from_lines(edges, scale, frame.shape)
        if line_result is not None:
            return line_result

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (small.shape[0] * small.shape[1]) * self.config.min_area_ratio
        best_rect = None
        best_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(contour)
            (w, h) = rect[1]
            if w <= 1 or h <= 1:
                continue
            ratio = max(w, h) / max(1.0, min(w, h))
            if ratio < self.config.min_aspect:
                continue
            score = area * ratio
            if score > best_score:
                best_score = score
                best_rect = rect

        if best_rect is None:
            return None

        box = cv2.boxPoints(best_rect)
        polygon = (box / scale).astype(np.float32)
        origin, length_dir, width_dir, length, board_width = _compute_axes(polygon)

        mask = np.zeros(edges.shape, dtype=np.uint8)
        box_small = np.array(box, dtype=np.int32)
        cv2.fillConvexPoly(mask, box_small, 255)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

        lines = cv2.HoughLinesP(
            masked_edges,
            1,
            math.pi / 180.0,
            threshold=60,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )
        string_positions: List[float] = []
        fret_positions: List[float] = []
        string_mids: List[Tuple[float, float]] = []
        fret_mids: List[Tuple[float, float]] = []
        line_points: List[Tuple[float, float]] = []
        if lines is not None:
            board_angle = _normalize_angle(math.degrees(math.atan2(length_dir[1], length_dir[0])))
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [float(val) / scale for val in line]
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                    continue
                angle = _normalize_angle(math.degrees(math.atan2(dy, dx)))
                diff = abs(angle - board_angle)
                diff = min(diff, 180.0 - diff)
                mid = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                rel = (mid[0] - origin[0], mid[1] - origin[1])
                if diff <= self.config.angle_tol:
                    pos = rel[0] * width_dir[0] + rel[1] * width_dir[1]
                    if 0 <= pos <= board_width:
                        string_positions.append(pos)
                        string_mids.append(mid)
                        line_points.append((x1, y1))
                        line_points.append((x2, y2))
                elif abs(diff - 90.0) <= self.config.angle_tol:
                    pos = rel[0] * length_dir[0] + rel[1] * length_dir[1]
                    if 0 <= pos <= length:
                        fret_positions.append(pos)
                        fret_mids.append(mid)
                        line_points.append((x1, y1))
                        line_points.append((x2, y2))

        if len(line_points) >= 4:
            proj_len_all = _project_values(line_points, length_dir)
            proj_w_all = _project_values(line_points, width_dir)
            min_len_all, max_len_all = min(proj_len_all), max(proj_len_all)
            min_w_all, max_w_all = min(proj_w_all), max(proj_w_all)
            proj_len_frets = _project_values(fret_mids, length_dir)
            proj_w_strings = _project_values(string_mids, width_dir)

            if len(proj_len_frets) >= self.config.min_fret_lines:
                min_len, max_len = _trim_bounds(proj_len_frets, self.config.line_trim_ratio)
            else:
                min_len, max_len = min_len_all, max_len_all

            if len(proj_w_strings) >= self.config.min_string_lines:
                min_w, max_w = _trim_bounds(proj_w_strings, self.config.line_trim_ratio)
            else:
                min_w, max_w = min_w_all, max_w_all

            ref_length = max_len - min_len
            ref_width = max_w - min_w
            ratio = max(ref_length, ref_width) / max(1.0, min(ref_length, ref_width))
            if ref_length > 1.0 and ref_width > 1.0 and ratio >= self.config.min_aspect * 0.6:
                pad_len = max(6.0, ref_length * self.config.fret_pad_ratio)
                pad_w = max(6.0, ref_width * self.config.string_pad_ratio)
                min_len -= pad_len
                max_len += pad_len
                min_w -= pad_w
                max_w += pad_w
                origin = (
                    float(length_dir[0] * min_len + width_dir[0] * min_w),
                    float(length_dir[1] * min_len + width_dir[1] * min_w),
                )
                length = max_len - min_len
                board_width = max_w - min_w
                polygon = _rect_from_axes(origin, length_dir, width_dir, length, board_width)

        if string_mids or fret_mids:
            string_positions = []
            for mid in string_mids:
                rel = (mid[0] - origin[0], mid[1] - origin[1])
                pos = rel[0] * width_dir[0] + rel[1] * width_dir[1]
                if 0 <= pos <= board_width:
                    string_positions.append(pos)
            fret_positions = []
            for mid in fret_mids:
                rel = (mid[0] - origin[0], mid[1] - origin[1])
                pos = rel[0] * length_dir[0] + rel[1] * length_dir[1]
                if 0 <= pos <= length:
                    fret_positions.append(pos)

        string_tol = max(4.0, board_width * self.config.string_cluster_ratio)
        fret_tol = max(6.0, length * self.config.fret_cluster_ratio)
        string_positions = _cluster_positions(string_positions, string_tol)
        fret_positions = _cluster_positions(fret_positions, fret_tol)
        string_positions.sort()
        fret_positions.sort()

        if len(string_positions) < max(3, self.config.string_count // 2):
            step = board_width / max(1, self.config.string_count - 1)
            string_positions = [idx * step for idx in range(self.config.string_count)]

        origin_is_nut = True
        fret_positions_from_nut = list(fret_positions)
        if len(fret_positions) >= 2:
            first_gap = fret_positions[1] - fret_positions[0]
            last_gap = fret_positions[-1] - fret_positions[-2]
            origin_is_nut = first_gap > last_gap
            if not origin_is_nut:
                fret_positions_from_nut = [length - pos for pos in reversed(fret_positions)]

        return FretboardResult(
            polygon=polygon,
            origin=origin,
            length_dir=length_dir,
            width_dir=width_dir,
            length=length,
            width=board_width,
            string_positions=string_positions,
            fret_positions=fret_positions,
            fret_positions_from_nut=fret_positions_from_nut,
            origin_is_nut=origin_is_nut,
        )


class FretboardTracker:
    def __init__(self, config: Optional[FretboardConfig] = None) -> None:
        self.config = config or FretboardConfig()
        self._detector = FretboardDetector(self.config)
        self._last: Optional[FretboardResult] = None
        self._last_time = 0.0

    def update(
        self, frame, now: Optional[float] = None, hands=None
    ) -> Optional[FretboardResult]:
        if now is None:
            now = time.time()
        exclude_mask = None
        if self.config.mask_exclude_hands and hands:
            exclude_mask = self._build_hand_mask(hands, frame.shape)
        result = self._detector.detect(frame, exclude_mask=exclude_mask)
        if result is not None:
            if self._last is not None and self.config.smooth_alpha > 0:
                alpha = max(0.0, min(1.0, self.config.smooth_alpha))
                blended = result.polygon * alpha + self._last.polygon * (1.0 - alpha)
                origin, length_dir, width_dir, length, board_width = _compute_axes(blended)
                result = FretboardResult(
                    polygon=blended,
                    origin=origin,
                    length_dir=length_dir,
                    width_dir=width_dir,
                    length=length,
                    width=board_width,
                    string_positions=result.string_positions,
                    fret_positions=result.fret_positions,
                    fret_positions_from_nut=result.fret_positions_from_nut,
                    origin_is_nut=result.origin_is_nut,
                )
            if hands:
                result = self._refine_with_hands(result, hands)
            self._last = result
            self._last_time = now
            return result
        if self._last is not None and now - self._last_time <= self.config.hold_seconds:
            if hands:
                return self._refine_with_hands(self._last, hands)
            return self._last
        return None

    def build_mask(
        self,
        frame,
        hands=None,
        include_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        exclude_mask = None
        if self.config.mask_exclude_hands and hands:
            exclude_mask = self._build_hand_mask(hands, frame.shape)
        return self._detector.build_mask(
            frame, include_mask=include_mask, exclude_mask=exclude_mask
        )

    def _build_hand_mask(self, hands, frame_shape) -> np.ndarray:
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        radius = max(6, int(min(height, width) * 0.015))
        for hand in hands:
            hull = getattr(hand, "hull", None)
            if hull is not None and len(hull) >= 3:
                cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
                continue
            points = getattr(hand, "landmarks_2d", None)
            if points:
                for point in points:
                    cv2.circle(
                        mask,
                        (int(point[0]), int(point[1])),
                        radius,
                        255,
                        -1,
                    )
                continue
            center = getattr(hand, "center", None)
            if center is not None:
                cv2.circle(mask, (int(center[0]), int(center[1])), radius * 2, 255, -1)

        dilate = max(0, int(self.config.mask_hand_dilate))
        if dilate > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilate)
        return mask

    def _refine_with_hands(
        self, result: FretboardResult, hands
    ) -> FretboardResult:
        if result is None:
            return result
        polygon = result.polygon.astype(np.int32)
        margin = max(18.0, result.width * self.config.hand_refine_margin_ratio)
        best_points: List[Tuple[float, float]] = []

        for hand in hands:
            points = getattr(hand, "landmarks_2d", None)
            if points is None or len(points) < 21:
                continue
            nearby: List[Tuple[float, float]] = []
            for idx in FINGER_TIPS.values():
                tip = points[idx]
                dist = cv2.pointPolygonTest(
                    polygon, (float(tip[0]), float(tip[1])), True
                )
                if dist >= -margin:
                    nearby.append((float(tip[0]), float(tip[1])))
            if len(nearby) > len(best_points):
                best_points = nearby

        if len(best_points) < self.config.hand_refine_min_points:
            return result

        origin = result.origin
        length_dir = result.length_dir
        width_dir = result.width_dir
        length = result.length
        board_width = result.width

        pos_len = [
            (pt[0] - origin[0]) * length_dir[0]
            + (pt[1] - origin[1]) * length_dir[1]
            for pt in best_points
        ]
        pos_w = [
            (pt[0] - origin[0]) * width_dir[0]
            + (pt[1] - origin[1]) * width_dir[1]
            for pt in best_points
        ]
        if not pos_len or not pos_w:
            return result

        len_lower = max(pos - length for pos in pos_len)
        len_upper = min(pos for pos in pos_len)
        if len_lower <= len_upper:
            delta_len = (len_lower + len_upper) * 0.5
        else:
            mean_len = sum(pos_len) / len(pos_len)
            delta_len = mean_len - length * 0.5

        w_lower = max(pos - board_width for pos in pos_w)
        w_upper = min(pos for pos in pos_w)
        if w_lower <= w_upper:
            delta_w = (w_lower + w_upper) * 0.5
        else:
            mean_w = sum(pos_w) / len(pos_w)
            delta_w = mean_w - board_width * 0.5

        max_shift_len = length * self.config.hand_refine_shift_len_ratio
        max_shift_w = board_width * self.config.hand_refine_shift_width_ratio
        delta_len = max(-max_shift_len, min(max_shift_len, delta_len))
        delta_w = max(-max_shift_w, min(max_shift_w, delta_w))

        if abs(delta_len) < 1e-3 and abs(delta_w) < 1e-3:
            return result

        shift_x = length_dir[0] * delta_len + width_dir[0] * delta_w
        shift_y = length_dir[1] * delta_len + width_dir[1] * delta_w
        origin = (origin[0] + shift_x, origin[1] + shift_y)
        polygon = _rect_from_axes(origin, length_dir, width_dir, length, board_width)

        return FretboardResult(
            polygon=polygon,
            origin=origin,
            length_dir=length_dir,
            width_dir=width_dir,
            length=length,
            width=board_width,
            string_positions=result.string_positions,
            fret_positions=result.fret_positions,
            fret_positions_from_nut=result.fret_positions_from_nut,
            origin_is_nut=result.origin_is_nut,
        )

    def locate_fingers(self, hands, result: FretboardResult) -> List[FingerPlacement]:
        placements: List[FingerPlacement] = []
        if result is None:
            return placements
        polygon = result.polygon.astype(np.int32)
        origin = result.origin
        length_dir = result.length_dir
        width_dir = result.width_dir
        length = result.length
        width = result.width
        string_positions = result.string_positions
        frets = result.fret_positions_from_nut
        for hand in hands:
            points = getattr(hand, "landmarks_2d", None)
            if points is None or len(points) < 21:
                continue
            for name, idx in FINGER_TIPS.items():
                tip = points[idx]
                if cv2.pointPolygonTest(polygon, (float(tip[0]), float(tip[1])), False) < 0:
                    continue
                rel = (tip[0] - origin[0], tip[1] - origin[1])
                pos_len = rel[0] * length_dir[0] + rel[1] * length_dir[1]
                pos_w = rel[0] * width_dir[0] + rel[1] * width_dir[1]
                if pos_len < 0 or pos_len > length or pos_w < 0 or pos_w > width:
                    continue
                pos_len_from_nut = pos_len if result.origin_is_nut else (length - pos_len)

                string_index = None
                if string_positions:
                    nearest = min(
                        range(len(string_positions)),
                        key=lambda i: abs(string_positions[i] - pos_w),
                    )
                    string_index = nearest + 1

                fret_index = None
                if frets:
                    for idx_fret, fret_pos in enumerate(frets, start=1):
                        if pos_len_from_nut < fret_pos:
                            fret_index = idx_fret
                            break
                    if fret_index is None:
                        fret_index = len(frets) + 1

                placements.append(
                    FingerPlacement(
                        name=name,
                        point=(int(tip[0]), int(tip[1])),
                        string_index=string_index,
                        fret_index=fret_index,
                    )
                )
        return placements


def draw_fretboard(frame, result: FretboardResult, placements: List[FingerPlacement]) -> None:
    if result is None:
        return
    poly = result.polygon.astype(np.int32)
    cv2.polylines(frame, [poly], True, IOS_BLUE, 2)

    origin = result.origin
    length_dir = result.length_dir
    width_dir = result.width_dir
    length = result.length
    width = result.width

    for pos in result.string_positions:
        start = (
            int(origin[0] + width_dir[0] * pos),
            int(origin[1] + width_dir[1] * pos),
        )
        end = (
            int(origin[0] + width_dir[0] * pos + length_dir[0] * length),
            int(origin[1] + width_dir[1] * pos + length_dir[1] * length),
        )
        cv2.line(frame, start, end, IOS_BLUE_SOFT, 1)

    for pos in result.fret_positions:
        start = (
            int(origin[0] + length_dir[0] * pos),
            int(origin[1] + length_dir[1] * pos),
        )
        end = (
            int(origin[0] + length_dir[0] * pos + width_dir[0] * width),
            int(origin[1] + length_dir[1] * pos + width_dir[1] * width),
        )
        cv2.line(frame, start, end, IOS_BORDER, 1)

    for placement in placements:
        label = placement.name
        if placement.string_index is not None:
            label += f"S{placement.string_index}"
        if placement.fret_index is not None:
            label += f"F{placement.fret_index}"
        draw_text(
            frame,
            label,
            (placement.point[0] + 6, placement.point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            IOS_TEXT,
            1,
            shadow=False,
        )
