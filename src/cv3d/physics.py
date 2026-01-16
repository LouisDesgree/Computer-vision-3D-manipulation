from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .hand_input import HandState


@dataclass
class PhysicsConfig:
    contact_force: float = 900.0
    hand_velocity_scale: float = 0.7
    contact_distance: float = 8.0
    gravity: float = 1600.0
    damping: float = 0.92
    restitution: float = 0.86
    max_speed: float = 1400.0
    rotation_smoothing: float = 0.2
    spin_strength: float = 0.2
    scale_smoothing: float = 0.35
    two_hand_min_scale: float = 0.7
    two_hand_max_scale: float = 1.6
    two_hand_squash_power: float = 0.6
    depth_scale_strength: float = 6.0
    depth_min_scale: float = 0.6
    depth_max_scale: float = 2.2
    enable_grab: bool = True
    grab_distance: float = 120.0
    grab_strength: float = 30.0
    grab_damping: float = 0.85
    grab_follow: float = 0.7


@dataclass
class CubeState:
    position: Optional[Tuple[float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    yaw: float = 20.0
    pitch: float = -15.0
    yaw_target: float = 20.0
    pitch_target: float = -15.0
    grabbed_by: Optional[int] = None
    grab_is_grip: bool = False
    grab_offset: Tuple[float, float] = (0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    scale_target: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    two_hand_base_dist: float = 0.0
    grab_depth: Optional[float] = None
    last_time: float = 0.0


def compute_dt(state: CubeState, now: float) -> float:
    if state.last_time == 0.0:
        dt = 1 / 30
    else:
        dt = min(max(now - state.last_time, 1 / 120), 0.05)
    state.last_time = now
    return dt


def _lerp(current: float, target: float, alpha: float) -> float:
    return current + alpha * (target - current)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hand_depth(hand: "HandState") -> Optional[float]:
    landmarks = getattr(hand, "landmarks_3d", None)
    if not landmarks or len(landmarks) <= 17:
        return None
    palm_indices = (0, 5, 9, 13, 17)
    return sum(landmarks[idx][2] for idx in palm_indices) / len(palm_indices)


class CubePhysics:
    def __init__(self, config: PhysicsConfig) -> None:
        self.config = config

    def step(
        self,
        state: CubeState,
        hands: List["HandState"],
        contact_infos: List[Optional[Tuple[float, Tuple[float, float]]]],
        dt: float,
    ) -> bool:
        if state.position is None:
            return False

        vel_x, vel_y = state.velocity
        contact = False
        grab_hand: Optional["HandState"] = None
        two_hand = False
        two_hand_mid = None
        two_hand_velocity = None
        two_hand_dist = None
        correction_x = 0.0
        correction_y = 0.0

        if self.config.enable_grab:
            if state.grabbed_by is not None:
                for hand in hands:
                    if hand.id == state.grabbed_by and hand.pinch:
                        grab_hand = hand
                        state.grab_is_grip = False
                        break
                if grab_hand is None:
                    state.grabbed_by = None
                    state.grab_is_grip = False
                    state.grab_depth = None

            pinch_candidates: List[Tuple[float, "HandState"]] = []
            cube_x, cube_y = state.position
            for hand, info in zip(hands, contact_infos):
                dx = cube_x - hand.center[0]
                dy = cube_y - hand.center[1]
                center_dist = math.hypot(dx, dy)
                touching = info is not None and info[0] <= 0
                pinch_active = hand.pinch and hand.pinch_strength >= 0.6
                if pinch_active and touching:
                    pinch_candidates.append((center_dist, hand))

            if len(pinch_candidates) >= 2:
                pinch_candidates.sort(key=lambda item: item[0])
                hand_a = pinch_candidates[0][1]
                hand_b = pinch_candidates[1][1]
                two_hand = True
                two_hand_mid = (
                    (hand_a.center[0] + hand_b.center[0]) / 2.0,
                    (hand_a.center[1] + hand_b.center[1]) / 2.0,
                )
                two_hand_velocity = (
                    (hand_a.velocity[0] + hand_b.velocity[0]) / 2.0,
                    (hand_a.velocity[1] + hand_b.velocity[1]) / 2.0,
                )
                two_hand_dist = math.hypot(
                    hand_a.center[0] - hand_b.center[0],
                    hand_a.center[1] - hand_b.center[1],
                )
                state.grabbed_by = None
                state.grab_is_grip = False
                state.grab_offset = (0.0, 0.0)
                state.grab_depth = None
            elif grab_hand is None:
                if pinch_candidates:
                    pinch_candidates.sort(key=lambda item: item[0])
                    grab_hand = pinch_candidates[0][1]
                    state.grabbed_by = grab_hand.id
                    state.grab_is_grip = False
                if grab_hand is not None:
                    state.grab_offset = (
                        state.position[0] - grab_hand.center[0],
                        state.position[1] - grab_hand.center[1],
                    )
                    state.grab_depth = _hand_depth(grab_hand)

            if not two_hand:
                state.two_hand_base_dist = 0.0
        else:
            if state.grabbed_by is not None:
                state.grabbed_by = None
                state.grab_is_grip = False
                state.grab_offset = (0.0, 0.0)
                state.grab_depth = None
            state.two_hand_base_dist = 0.0
            state.scale_target = (1.0, 1.0, 1.0)

        if two_hand and two_hand_mid is not None and two_hand_velocity is not None:
            contact = True
            if state.two_hand_base_dist <= 0.0:
                state.two_hand_base_dist = max(two_hand_dist or 1.0, 1.0)
            ratio = (two_hand_dist or state.two_hand_base_dist) / state.two_hand_base_dist
            ratio = _clamp(ratio, self.config.two_hand_min_scale, self.config.two_hand_max_scale)
            squash = ratio ** -self.config.two_hand_squash_power
            state.scale_target = (ratio, squash, squash)

            delta_x = two_hand_mid[0] - state.position[0]
            delta_y = two_hand_mid[1] - state.position[1]
            vel_x += delta_x * self.config.grab_strength * 1.2 * dt
            vel_y += delta_y * self.config.grab_strength * 1.2 * dt
            vel_x = vel_x * self.config.grab_damping + two_hand_velocity[0] * self.config.grab_follow
            vel_y = vel_y * self.config.grab_damping + two_hand_velocity[1] * self.config.grab_follow
        elif grab_hand is not None:
            contact = True
            target_x = grab_hand.center[0] + state.grab_offset[0]
            target_y = grab_hand.center[1] + state.grab_offset[1]
            if state.grab_depth is None:
                state.grab_depth = _hand_depth(grab_hand)
            depth_now = _hand_depth(grab_hand)
            if depth_now is not None and state.grab_depth is not None:
                delta = state.grab_depth - depth_now
                scale = 1.0 + delta * self.config.depth_scale_strength
                scale = _clamp(scale, self.config.depth_min_scale, self.config.depth_max_scale)
                state.scale_target = (scale, scale, scale)
            if state.grab_is_grip and grab_hand.grip:
                vel_x, vel_y = grab_hand.velocity
                if dt > 0:
                    state.position = (target_x - vel_x * dt, target_y - vel_y * dt)
            else:
                delta_x = target_x - state.position[0]
                delta_y = target_y - state.position[1]
                vel_x += delta_x * self.config.grab_strength * dt
                vel_y += delta_y * self.config.grab_strength * dt
                vel_x = (
                    vel_x * self.config.grab_damping + grab_hand.velocity[0] * self.config.grab_follow
                )
                vel_y = (
                    vel_y * self.config.grab_damping + grab_hand.velocity[1] * self.config.grab_follow
                )
        else:
            for hand, info in zip(hands, contact_infos):
                if info is None:
                    continue
                distance, normal = info
                if distance > self.config.contact_distance:
                    continue
                contact = True
                vel_dot = vel_x * normal[0] + vel_y * normal[1]
                penetration = max(0.0, -distance)
                if penetration > 0 and self.config.contact_distance > 0:
                    correction = min(penetration, self.config.contact_distance) * 0.6
                    correction_x += normal[0] * correction
                    correction_y += normal[1] * correction
                if penetration > 0 or vel_dot < 0:
                    push_scale = 0.35
                    if self.config.contact_distance > 0:
                        if distance > 0:
                            push_scale *= (self.config.contact_distance - distance) / self.config.contact_distance
                        else:
                            push_scale = 1.0 + min(penetration / self.config.contact_distance, 1.0)
                    vel_x += normal[0] * self.config.contact_force * push_scale * dt
                    vel_y += normal[1] * self.config.contact_force * push_scale * dt

            vel_y += self.config.gravity * dt
            damping = self.config.damping ** (dt * 60.0)
            vel_x *= damping
            vel_y *= damping
            state.grab_depth = None

        if two_hand:
            state.two_hand_base_dist = max(state.two_hand_base_dist, 1.0)

        speed = math.hypot(vel_x, vel_y)
        if speed > self.config.max_speed and speed > 0:
            scale = self.config.max_speed / speed
            vel_x *= scale
            vel_y *= scale

        pos_x, pos_y = state.position
        pos_x += vel_x * dt + correction_x
        pos_y += vel_y * dt + correction_y
        state.position = (pos_x, pos_y)
        state.velocity = (vel_x, vel_y)

        state.yaw_target += vel_x * self.config.spin_strength * dt
        state.pitch_target -= vel_y * self.config.spin_strength * dt
        state.pitch_target = _clamp(state.pitch_target, -80.0, 80.0)
        state.yaw = _lerp(state.yaw, state.yaw_target, self.config.rotation_smoothing)
        state.pitch = _lerp(state.pitch, state.pitch_target, self.config.rotation_smoothing)

        scale_x, scale_y, scale_z = state.scale
        target_x, target_y, target_z = state.scale_target
        state.scale = (
            _lerp(scale_x, target_x, self.config.scale_smoothing),
            _lerp(scale_y, target_y, self.config.scale_smoothing),
            _lerp(scale_z, target_z, self.config.scale_smoothing),
        )

        return contact

    def apply_bounds(
        self,
        state: CubeState,
        width: int,
        height: int,
        half_w: float,
        half_h: float,
    ) -> None:
        if state.position is None:
            return
        pos_x, pos_y = state.position
        vel_x, vel_y = state.velocity
        hit_x = False
        hit_y = False

        if pos_x - half_w < 0:
            pos_x = half_w
            vel_x = abs(vel_x) * self.config.restitution
            hit_x = True
        elif pos_x + half_w > width:
            pos_x = width - half_w
            vel_x = -abs(vel_x) * self.config.restitution
            hit_x = True

        if pos_y - half_h < 0:
            pos_y = half_h
            vel_y = abs(vel_y) * self.config.restitution
            hit_y = True
        elif pos_y + half_h > height:
            pos_y = height - half_h
            vel_y = -abs(vel_y) * self.config.restitution
            hit_y = True

        state.position = (pos_x, pos_y)
        state.velocity = (vel_x, vel_y)
        if hit_x:
            state.yaw_target += vel_x * 0.015
        if hit_y:
            state.pitch_target -= vel_y * 0.015
