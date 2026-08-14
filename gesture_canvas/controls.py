"""Keyboard-controlled drawing settings."""

from __future__ import annotations


MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 30
DEFAULT_BRUSH_SIZE = 7


def adjust_brush_size(current: int, change: int) -> int:
    """Return a brush size clamped to the supported range."""

    return max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, current + change))
