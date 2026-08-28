"""Small dependency-free cursor smoothing utilities."""

from __future__ import annotations

from math import hypot


class ExponentialSmoother:
    """Smooth noisy cursor coordinates while keeping interaction responsive."""

    def __init__(self, alpha: float = 0.45) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._point: tuple[float, float] | None = None

    def update(self, point: tuple[int, int]) -> tuple[int, int]:
        if self._point is None:
            self._point = float(point[0]), float(point[1])
        else:
            x = self.alpha * point[0] + (1 - self.alpha) * self._point[0]
            y = self.alpha * point[1] + (1 - self.alpha) * self._point[1]
            self._point = x, y
        return round(self._point[0]), round(self._point[1])

    def reset(self) -> None:
        self._point = None


class AdaptiveSmoother:
    """Use strong smoothing for jitter and faster response for intentional motion."""

    def __init__(
        self,
        min_alpha: float = 0.24,
        max_alpha: float = 0.72,
        response_distance: float = 70.0,
    ) -> None:
        if not 0 < min_alpha <= max_alpha <= 1:
            raise ValueError("Expected 0 < min_alpha <= max_alpha <= 1")
        if response_distance <= 0:
            raise ValueError("response_distance must be positive")
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.response_distance = response_distance
        self._point: tuple[float, float] | None = None

    def update(self, point: tuple[int, int]) -> tuple[int, int]:
        if self._point is None:
            self._point = float(point[0]), float(point[1])
        else:
            distance = hypot(point[0] - self._point[0], point[1] - self._point[1])
            motion = min(1.0, distance / self.response_distance)
            alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * motion
            x = alpha * point[0] + (1 - alpha) * self._point[0]
            y = alpha * point[1] + (1 - alpha) * self._point[1]
            self._point = x, y
        return round(self._point[0]), round(self._point[1])

    def reset(self) -> None:
        self._point = None


class ScalarSmoother:
    """Smooth a noisy scalar signal such as apparent palm size."""

    def __init__(self, alpha: float = 0.22) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value: float | None = None

    def update(self, value: float) -> float:
        if value < 0:
            raise ValueError("value cannot be negative")
        if self._value is None:
            self._value = float(value)
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None
