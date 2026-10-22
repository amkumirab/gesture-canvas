"""Keyboard-controlled drawing settings."""

from __future__ import annotations


MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 30
DEFAULT_BRUSH_SIZE = 7


def adjust_brush_size(current: int, change: int) -> int:
    """Return a brush size clamped to the supported range."""

    return max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, current + change))


def transform_mode_for_key(key: int) -> str | None:
    """Map number shortcuts to spatial controls without taking drawing shortcuts."""

    return {
        ord("0"): "auto",
        ord("1"): "scale",
        ord("2"): "tilt_x",
        ord("3"): "tilt_y",
        ord("4"): "spin",
    }.get(key)
