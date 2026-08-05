"""Small dependency-free cursor smoothing utilities."""

from __future__ import annotations


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

