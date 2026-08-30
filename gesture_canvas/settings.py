"""Validated, persistent user settings for gesture interaction."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GestureSettings:
    schema_version: int = 1
    camera_index: int = 0
    pinch_close_ratio: float = 0.34
    pinch_release_ratio: float = 0.52
    draw_min_alpha: float = 0.24
    draw_max_alpha: float = 0.72
    pinch_min_alpha: float = 0.30
    pinch_max_alpha: float = 0.82
    depth_alpha: float = 0.22
    depth_sensitivity: float = 1.35
    selection_radius_ratio: float = 0.025
    calibrated: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported settings schema")
        if type(self.camera_index) is not int or self.camera_index < 0:
            raise ValueError("camera_index cannot be negative")
        for name in (
            "pinch_close_ratio",
            "pinch_release_ratio",
            "draw_min_alpha",
            "draw_max_alpha",
            "pinch_min_alpha",
            "pinch_max_alpha",
            "depth_alpha",
            "depth_sensitivity",
            "selection_radius_ratio",
        ):
            _validate_number(getattr(self, name), name)
        if type(self.calibrated) is not bool:
            raise ValueError("calibrated must be a boolean")
        if not 0 < self.pinch_close_ratio < self.pinch_release_ratio < 1:
            raise ValueError("Invalid pinch thresholds")
        _validate_alpha_pair(
            self.draw_min_alpha,
            self.draw_max_alpha,
            "draw",
        )
        _validate_alpha_pair(
            self.pinch_min_alpha,
            self.pinch_max_alpha,
            "pinch",
        )
        if not 0 < self.depth_alpha <= 1:
            raise ValueError("depth_alpha must be in (0, 1]")
        if not 0.5 <= self.depth_sensitivity <= 2.5:
            raise ValueError("depth_sensitivity must be between 0.5 and 2.5")
        if not 0.005 <= self.selection_radius_ratio <= 0.08:
            raise ValueError("selection_radius_ratio must be between 0.005 and 0.08")

    def selection_radius(self, frame_width: int) -> int:
        if frame_width <= 0:
            raise ValueError("frame_width must be positive")
        return max(18, min(56, round(frame_width * self.selection_radius_ratio)))

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: object) -> GestureSettings:
        if not isinstance(values, dict):
            raise ValueError("Settings must be a JSON object")
        defaults = cls().to_dict()
        known = {key: values.get(key, default) for key, default in defaults.items()}
        return cls(**known)


def default_settings_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / ".config"
    return root / "gesture-canvas" / "settings.json"


def load_settings(path: Path | None = None) -> GestureSettings:
    source = path or default_settings_path()
    try:
        if not source.exists():
            return GestureSettings()
        return GestureSettings.from_dict(
            json.loads(source.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return GestureSettings()


def save_settings(
    settings: GestureSettings,
    path: Path | None = None,
) -> Path:
    destination = path or default_settings_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(settings.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _validate_alpha_pair(minimum: float, maximum: float, name: str) -> None:
    if not 0 < minimum <= maximum <= 1:
        raise ValueError(f"Invalid {name} smoothing range")


def _validate_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
