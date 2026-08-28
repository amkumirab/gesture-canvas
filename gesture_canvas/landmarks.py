"""Hand landmark types and coordinate conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
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


def palm_span(landmarks: Sequence[Landmark]) -> float:
    """Return the apparent palm width used as a relative depth signal."""

    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}")
    return hypot(
        landmarks[5].x - landmarks[17].x,
        landmarks[5].y - landmarks[17].y,
    )


def pinch_point(landmarks: Sequence[Landmark]) -> Landmark:
    """Return the midpoint between thumb and index tips.

    The midpoint stays much steadier than either fingertip while a pinch is held,
    so manipulation does not jump when the two fingertips trade positions.
    """

    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}")
    thumb = landmarks[4]
    index = landmarks[8]
    return Landmark(
        (thumb.x + index.x) / 2,
        (thumb.y + index.y) / 2,
        (thumb.z + index.z) / 2,
    )


def to_pixel(point: Landmark, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized point to a clamped image coordinate."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x = int(np.clip(round(point.x * (width - 1)), 0, width - 1))
    y = int(np.clip(round(point.y * (height - 1)), 0, height - 1))
    return x, y
