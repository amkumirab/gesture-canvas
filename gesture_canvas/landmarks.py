"""Hand landmark types and coordinate conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Landmark:
    """One normalized MediaPipe hand landmark."""

    x: float
    y: float
    z: float = 0.0


def as_landmarks(raw_landmarks: Sequence[object]) -> list[Landmark]:
    """Convert MediaPipe-like objects into testable project landmarks."""

    if len(raw_landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(raw_landmarks)}")
    return [Landmark(float(p.x), float(p.y), float(p.z)) for p in raw_landmarks]


def to_pixel(point: Landmark, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized point to a clamped image coordinate."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x = int(np.clip(round(point.x * (width - 1)), 0, width - 1))
    y = int(np.clip(round(point.y * (height - 1)), 0, height - 1))
    return x, y
