"""Hand landmark types, normalization, and feature extraction."""

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


def normalized_features(landmarks: Sequence[Landmark]) -> np.ndarray:
    """Return translation- and scale-invariant 63-value hand features.

    Coordinates are centered on the wrist and scaled using the greatest 3D
    distance from it. This makes samples more robust to camera distance and
    position while preserving the hand pose.
    """

    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}")

    points = np.asarray([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
    points -= points[0]
    scale = float(np.linalg.norm(points, axis=1).max())
    if scale < 1e-6:
        raise ValueError("Cannot normalize collapsed hand landmarks")
    return (points / scale).reshape(-1)


def to_pixel(point: Landmark, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized point to a clamped image coordinate."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x = int(np.clip(round(point.x * (width - 1)), 0, width - 1))
    y = int(np.clip(round(point.y * (height - 1)), 0, height - 1))
    return x, y

