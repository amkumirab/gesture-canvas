"""Deterministic hand-pose recognition rules."""

from __future__ import annotations

from enum import Enum
from math import hypot, sqrt
from typing import Sequence

from .landmarks import Landmark


class Gesture(str, Enum):
    IDLE = "idle"
    DRAW = "draw"
    ERASE = "erase"
    PINCH = "pinch"
    OPEN_PALM = "open_palm"


FINGER_JOINTS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


class PinchDetector:
    """Scale-independent pinch detector with temporal and distance hysteresis."""

    def __init__(
        self,
        close_ratio: float = 0.34,
        release_ratio: float = 0.52,
        close_frames: int = 2,
        release_frames: int = 2,
    ) -> None:
        if not 0 < close_ratio < release_ratio < 1:
            raise ValueError("Expected 0 < close_ratio < release_ratio < 1")
        if close_frames < 1 or release_frames < 1:
            raise ValueError("Pinch confirmation frame counts must be positive")
        self.close_ratio = close_ratio
        self.release_ratio = release_ratio
        self.close_frames = close_frames
        self.release_frames = release_frames
        self.active = False
        self._close_count = 0
        self._release_count = 0

    def update(
        self,
        landmarks: Sequence[Landmark],
    ) -> tuple[bool, bool, bool, float]:
        """Return ``active, started, ended, ratio`` for the current frame."""

        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}")
        palm_width = _distance_3d(landmarks[5], landmarks[17])
        if palm_width <= 1e-6:
            was_active = self.active
            self.reset()
            return False, False, was_active, float("inf")

        ratio = _distance_3d(landmarks[4], landmarks[8]) / palm_width
        was_active = self.active
        if self.active:
            self._close_count = 0
            if ratio >= self.release_ratio:
                self._release_count += 1
                if self._release_count >= self.release_frames:
                    self.active = False
                    self._release_count = 0
            else:
                self._release_count = 0
        else:
            self._release_count = 0
            if ratio < self.close_ratio:
                self._close_count += 1
                if self._close_count >= self.close_frames:
                    self.active = True
                    self._close_count = 0
            else:
                self._close_count = 0
        return self.active, self.active and not was_active, was_active and not self.active, ratio

    def reset(self) -> None:
        self.active = False
        self._close_count = 0
        self._release_count = 0


class DrawGestureStabilizer:
    """Bridge brief tracking dropouts without delaying deliberate tools."""

    def __init__(self, grace_seconds: float = 0.18, erase_confirm_frames: int = 2) -> None:
        if grace_seconds < 0:
            raise ValueError("grace_seconds cannot be negative")
        if erase_confirm_frames < 1:
            raise ValueError("erase_confirm_frames must be positive")
        self.grace_seconds = grace_seconds
        self.erase_confirm_frames = erase_confirm_frames
        self._last_draw_at: float | None = None
        self._erase_count = 0

    def update(self, gesture: Gesture, now: float) -> Gesture:
        if gesture is Gesture.DRAW:
            self._erase_count = 0
            self._last_draw_at = now
            return gesture
        if gesture is Gesture.ERASE:
            self._erase_count += 1
            if self._erase_count >= self.erase_confirm_frames:
                self._last_draw_at = None
                return gesture
            return Gesture.DRAW if self.hold_during_missing(now) else Gesture.IDLE
        self._erase_count = 0
        if gesture is Gesture.PINCH:
            self.reset()
            return gesture
        if self.hold_during_missing(now):
            return Gesture.DRAW
        return gesture

    def hold_during_missing(self, now: float) -> bool:
        if self._last_draw_at is None:
            return False
        if now - self._last_draw_at <= self.grace_seconds:
            return True
        self.reset()
        return False

    def reset(self) -> None:
        self._last_draw_at = None
        self._erase_count = 0


class GestureRecognizer:
    """Recognize supported hand poses using transparent geometry rules."""

    def __init__(self, pinch_ratio: float = 0.32) -> None:
        if not 0 < pinch_ratio < 1:
            raise ValueError("pinch_ratio must be between zero and one")
        self.pinch_ratio = pinch_ratio

    def recognize(self, landmarks: Sequence[Landmark]) -> tuple[Gesture, float]:
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}")

        palm_width = self._distance(landmarks[5], landmarks[17])
        pinch_distance = self._distance(landmarks[4], landmarks[8])
        if palm_width > 1e-6 and pinch_distance / palm_width < self.pinch_ratio:
            confidence = min(1.0, 1.0 - (pinch_distance / palm_width) / self.pinch_ratio)
            return Gesture.PINCH, max(0.55, confidence)

        extended = {
            name: self._finger_extended(landmarks, mcp, pip, tip)
            for name, (mcp, pip, tip) in FINGER_JOINTS.items()
        }
        pattern = tuple(extended.values())

        if pattern == (True, False, False, False):
            return Gesture.DRAW, 0.9
        if pattern == (True, True, False, False):
            return Gesture.ERASE, 0.9
        if pattern == (True, True, True, True):
            return Gesture.OPEN_PALM, 0.85
        return Gesture.IDLE, 0.65

    @staticmethod
    def _distance(a: Landmark, b: Landmark) -> float:
        return _distance(a, b)

    @staticmethod
    def _finger_extended(
        landmarks: Sequence[Landmark],
        mcp_index: int,
        pip_index: int,
        tip_index: int,
    ) -> bool:
        """Detect extension using finger geometry instead of screen direction."""

        mcp = landmarks[mcp_index]
        pip = landmarks[pip_index]
        tip = landmarks[tip_index]
        wrist = landmarks[0]
        first = (mcp.x - pip.x, mcp.y - pip.y)
        second = (tip.x - pip.x, tip.y - pip.y)
        first_norm = sqrt(sum(value * value for value in first))
        second_norm = sqrt(sum(value * value for value in second))
        if first_norm <= 1e-6 or second_norm <= 1e-6:
            return False
        cosine = sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
        reaches_past_pip = _distance_3d(tip, wrist) > _distance_3d(pip, wrist) * 1.05
        return cosine < -0.70 and reaches_past_pip


def _distance(a: Landmark, b: Landmark) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def _distance_3d(a: Landmark, b: Landmark) -> float:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
