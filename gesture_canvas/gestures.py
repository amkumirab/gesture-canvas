"""Deterministic gesture baseline used before a trained model is available."""

from __future__ import annotations

from enum import Enum
from math import hypot
from typing import Sequence

from .landmarks import Landmark


class Gesture(str, Enum):
    IDLE = "idle"
    DRAW = "draw"
    ERASE = "erase"
    PINCH = "pinch"
    OPEN_PALM = "open_palm"


FINGER_JOINTS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


class RuleBasedGestureClassifier:
    """Fast and explainable pose classifier for the MVP.

    The learned PyTorch classifier uses the same output labels and can replace
    this class at runtime without changing the drawing application.
    """

    def __init__(self, pinch_ratio: float = 0.32) -> None:
        if not 0 < pinch_ratio < 1:
            raise ValueError("pinch_ratio must be between zero and one")
        self.pinch_ratio = pinch_ratio

    def classify(self, landmarks: Sequence[Landmark]) -> tuple[Gesture, float]:
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}")

        palm_width = self._distance(landmarks[5], landmarks[17])
        pinch_distance = self._distance(landmarks[4], landmarks[8])
        if palm_width > 1e-6 and pinch_distance / palm_width < self.pinch_ratio:
            confidence = min(1.0, 1.0 - (pinch_distance / palm_width) / self.pinch_ratio)
            return Gesture.PINCH, max(0.55, confidence)

        extended = {
            name: landmarks[tip].y < landmarks[pip].y
            for name, (tip, pip) in FINGER_JOINTS.items()
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
        return hypot(a.x - b.x, a.y - b.y)

