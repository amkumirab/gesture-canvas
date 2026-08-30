"""Guided calibration for user-specific gesture thresholds and smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np

from .settings import GestureSettings


PREPARE_OPEN = "prepare_open"
CAPTURE_OPEN = "capture_open"
PREPARE_PINCH = "prepare_pinch"
CAPTURE_PINCH = "capture_pinch"
COMPLETE = "complete"
FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CalibrationMeasurement:
    pinch_ratio: float
    palm_span: float
    cursor: tuple[float, float]

    def __post_init__(self) -> None:
        if not isfinite(self.pinch_ratio) or self.pinch_ratio <= 0:
            raise ValueError("pinch_ratio must be positive and finite")
        if not isfinite(self.palm_span) or self.palm_span <= 0:
            raise ValueError("palm_span must be positive and finite")
        if not all(isfinite(value) for value in self.cursor):
            raise ValueError("cursor coordinates must be finite")


class CalibrationSession:
    """Collect open-hand and pinch samples, then derive stable settings."""

    def __init__(
        self,
        now: float,
        camera_index: int = 0,
        samples_per_phase: int = 36,
        preparation_seconds: float = 1.2,
    ) -> None:
        if camera_index < 0:
            raise ValueError("camera_index cannot be negative")
        if samples_per_phase < 3:
            raise ValueError("samples_per_phase must be at least three")
        if preparation_seconds < 0:
            raise ValueError("preparation_seconds cannot be negative")
        self.camera_index = camera_index
        self.samples_per_phase = samples_per_phase
        self.preparation_seconds = preparation_seconds
        self.stage = PREPARE_OPEN
        self.phase_started_at = float(now)
        self.open_samples: list[CalibrationMeasurement] = []
        self.pinch_samples: list[CalibrationMeasurement] = []
        self.result: GestureSettings | None = None
        self.error: str | None = None

    @property
    def is_active(self) -> bool:
        return self.stage not in {COMPLETE, FAILED}

    @property
    def progress(self) -> float:
        if self.stage in {PREPARE_OPEN, PREPARE_PINCH}:
            return 0.0
        if self.stage == CAPTURE_OPEN:
            return min(1.0, len(self.open_samples) / self.samples_per_phase)
        if self.stage == CAPTURE_PINCH:
            return min(1.0, len(self.pinch_samples) / self.samples_per_phase)
        return 1.0

    @property
    def instruction(self) -> str:
        return {
            PREPARE_OPEN: "OPEN THUMB AND INDEX - HOLD HAND STEADY",
            CAPTURE_OPEN: "KEEP THUMB AND INDEX OPEN",
            PREPARE_PINCH: "GET READY TO PINCH THUMB AND INDEX",
            CAPTURE_PINCH: "KEEP THUMB AND INDEX PINCHED",
            COMPLETE: "CALIBRATION SAVED",
            FAILED: self.error or "CALIBRATION FAILED",
        }[self.stage]

    def update(
        self,
        measurement: CalibrationMeasurement | None,
        now: float,
    ) -> GestureSettings | None:
        if not self.is_active:
            return self.result

        if self.stage in {PREPARE_OPEN, PREPARE_PINCH}:
            if now - self.phase_started_at < self.preparation_seconds:
                return None
            self.stage = (
                CAPTURE_OPEN if self.stage == PREPARE_OPEN else CAPTURE_PINCH
            )
            self.phase_started_at = float(now)
            return None

        if measurement is None:
            return None
        samples = (
            self.open_samples if self.stage == CAPTURE_OPEN else self.pinch_samples
        )
        samples.append(measurement)
        if len(samples) < self.samples_per_phase:
            return None

        if self.stage == CAPTURE_OPEN:
            self.stage = PREPARE_PINCH
            self.phase_started_at = float(now)
            return None

        try:
            self.result = _derive_settings(
                self.open_samples,
                self.pinch_samples,
                self.camera_index,
            )
        except ValueError as exc:
            self.stage = FAILED
            self.error = str(exc)
            return None
        self.stage = COMPLETE
        return self.result


def light_quality(brightness: float) -> str:
    if not isfinite(brightness):
        raise ValueError("brightness must be finite")
    if not 0 <= brightness <= 255:
        raise ValueError("brightness must be between zero and 255")
    if brightness < 48:
        return "LOW LIGHT"
    if brightness > 218:
        return "TOO BRIGHT"
    return "LIGHT OK"


def _derive_settings(
    open_samples: list[CalibrationMeasurement],
    pinch_samples: list[CalibrationMeasurement],
    camera_index: int,
) -> GestureSettings:
    if len(open_samples) < 3 or len(pinch_samples) < 3:
        raise ValueError("NOT ENOUGH HAND SAMPLES - PRESS K TO RETRY")

    open_ratio = float(np.median([sample.pinch_ratio for sample in open_samples]))
    pinch_ratio = float(np.median([sample.pinch_ratio for sample in pinch_samples]))
    separation = open_ratio - pinch_ratio
    if separation < 0.16:
        raise ValueError("OPEN AND PINCH WERE TOO SIMILAR - PRESS K TO RETRY")
    if pinch_ratio >= 0.44:
        raise ValueError("PINCH WAS NOT CLOSED ENOUGH - PRESS K TO RETRY")

    close_ratio = float(np.clip(pinch_ratio + separation * 0.22, 0.16, 0.46))
    if close_ratio <= pinch_ratio + 0.02:
        raise ValueError("PINCH RANGE WAS INVALID - PRESS K TO RETRY")
    release_ratio = float(
        np.clip(
            pinch_ratio + separation * 0.50,
            close_ratio + 0.08,
            0.82,
        )
    )
    if release_ratio <= close_ratio:
        raise ValueError("PINCH RANGE WAS INVALID - PRESS K TO RETRY")

    jitter = _median_cursor_step(open_samples + pinch_samples)
    draw_min_alpha = float(np.clip(0.30 - jitter * 12.0, 0.16, 0.30))
    pinch_min_alpha = float(np.clip(0.36 - jitter * 12.0, 0.20, 0.34))

    open_spans = np.asarray(
        [sample.palm_span for sample in open_samples],
        dtype=np.float64,
    )
    span_mean = max(float(open_spans.mean()), 1e-6)
    span_variation = float(open_spans.std() / span_mean)
    depth_alpha = float(np.clip(0.28 - span_variation * 1.8, 0.12, 0.28))
    depth_sensitivity = float(
        np.clip(1.40 - span_variation * 1.5, 1.05, 1.40)
    )
    selection_radius_ratio = float(
        np.clip(float(np.median(open_spans)) * 0.18, 0.018, 0.040)
    )

    return GestureSettings(
        camera_index=camera_index,
        pinch_close_ratio=close_ratio,
        pinch_release_ratio=release_ratio,
        draw_min_alpha=draw_min_alpha,
        draw_max_alpha=0.72,
        pinch_min_alpha=pinch_min_alpha,
        pinch_max_alpha=0.82,
        depth_alpha=depth_alpha,
        depth_sensitivity=depth_sensitivity,
        selection_radius_ratio=selection_radius_ratio,
        calibrated=True,
    )


def _median_cursor_step(samples: list[CalibrationMeasurement]) -> float:
    phase_steps: list[float] = []
    for first, second in zip(samples, samples[1:]):
        distance = hypot(
            second.cursor[0] - first.cursor[0],
            second.cursor[1] - first.cursor[1],
        )
        if distance < 0.08:
            phase_steps.append(distance)
    return float(np.median(phase_steps)) if phase_steps else 0.0
