"""On-screen instructions and quality feedback for gesture calibration."""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import (
    CAPTURE_OPEN,
    CAPTURE_PINCH,
    COMPLETE,
    FAILED,
    PREPARE_OPEN,
    PREPARE_PINCH,
    CalibrationSession,
    light_quality,
)


def draw_calibration_overlay(
    frame: np.ndarray,
    session: CalibrationSession,
    brightness: float,
    hand_confidence: float,
    camera_index: int,
) -> None:
    """Draw the full calibration card over a live frame."""

    height, width = frame.shape[:2]
    if height < 220 or width < 360:
        return

    panel_width = min(width - 32, 720)
    panel_height = 214
    left = (width - panel_width) // 2
    top = max(18, (height - panel_height) // 2)
    right = left + panel_width
    bottom = top + panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (14, 19, 27), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    accent = (70, 80, 245) if session.stage == FAILED else (80, 220, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), accent, 2)

    step = _step_label(session.stage)
    cv2.putText(
        frame,
        f"GESTURE CALIBRATION  |  {step}",
        (left + 22, top + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        accent,
        2,
        cv2.LINE_AA,
    )
    instruction_scale = 0.55 if panel_width >= 600 else 0.43
    cv2.putText(
        frame,
        session.instruction,
        (left + 22, top + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        instruction_scale,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )

    progress_left = left + 22
    progress_right = right - 22
    progress_top = top + 100
    progress_bottom = progress_top + 16
    cv2.rectangle(
        frame,
        (progress_left, progress_top),
        (progress_right, progress_bottom),
        (55, 62, 74),
        -1,
    )
    fill_right = progress_left + round(
        (progress_right - progress_left) * session.progress
    )
    if fill_right > progress_left:
        cv2.rectangle(
            frame,
            (progress_left, progress_top),
            (fill_right, progress_bottom),
            accent,
            -1,
        )

    light = light_quality(brightness)
    hand = "HAND OK" if hand_confidence >= 0.50 else "SHOW ONE HAND"
    quality_color = (
        (90, 220, 110)
        if light == "LIGHT OK" and hand_confidence >= 0.50
        else (60, 170, 255)
    )
    cv2.putText(
        frame,
        f"CAMERA {camera_index}   {light}   {hand} {hand_confidence:.0%}",
        (left + 22, top + 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        quality_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "K RESTART   ESC CANCEL   D RESTORE DEFAULTS",
        (left + 22, top + 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (185, 193, 205),
        1,
        cv2.LINE_AA,
    )


def _step_label(stage: str) -> str:
    if stage in {PREPARE_OPEN, CAPTURE_OPEN}:
        return "STEP 1 / 2"
    if stage in {PREPARE_PINCH, CAPTURE_PINCH}:
        return "STEP 2 / 2"
    if stage == COMPLETE:
        return "COMPLETE"
    return "TRY AGAIN"
