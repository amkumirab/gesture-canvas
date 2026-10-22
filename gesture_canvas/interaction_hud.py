"""Clear on-screen feedback for direct manipulation gestures."""

from __future__ import annotations

import cv2
import numpy as np


TRANSFORM_MODE_LABELS = {
    "auto": "AUTO",
    "scale": "SCALE",
    "tilt_x": "TILT X",
    "tilt_y": "TILT Y",
    "spin": "SPIN Z",
}


def draw_manipulation_hud(
    frame: np.ndarray,
    mode: str,
    transform_info: tuple[float, float, float, float] | None = None,
    recovering_tracking: bool = False,
    content_top: int = 74,
    active_control: str | None = None,
    requested_control: str = "auto",
) -> None:
    """Draw a compact gesture hint only while an object is being manipulated."""

    if mode == "idle" or frame.shape[0] < 130 or frame.shape[1] < 320:
        return
    if mode not in {"move", "move-3d", "arm-transform", "transform"}:
        raise ValueError(f"Unknown manipulation mode: {mode}")

    if mode == "arm-transform":
        title = "TWO-HAND READY"
        hint = (
            "HOLD BOTH PINCHES STEADY - THEN MOVE TO ROTATE"
            if requested_control == "auto"
            else f"{TRANSFORM_MODE_LABELS[requested_control]} READY - HOLD BOTH PINCHES"
        )
    elif mode == "transform":
        title = "TWO-HAND TRANSFORM"
        if active_control in TRANSFORM_MODE_LABELS and active_control != "auto":
            if requested_control == "auto":
                hint = (
                    f"{TRANSFORM_MODE_LABELS[active_control]} LOCKED   "
                    "RELEASE SECOND PINCH TO SWITCH"
                )
            else:
                hint = (
                    f"{TRANSFORM_MODE_LABELS[active_control]} MANUAL   "
                    "PRESS 0-4 TO SWITCH"
                )
        else:
            hint = "MOVE ONE WAY: APART=SCALE  TURN=SPIN  PAIR=TILT"
    elif mode == "move-3d":
        title = "3D MOVE"
        hint = "DRAG: X/Y   HAND CLOSER/FARTHER: Z"
    else:
        title = "MOVE"
        hint = "DRAG TO MOVE   OPEN PINCH TO DROP"

    height, width = frame.shape[:2]
    panel_width = min(width - 20, 570)
    left = max(10, (width - panel_width) // 2)
    right = left + panel_width
    top = min(max(content_top + 8, 8), height - 72)
    bottom = min(height - 8, top + 62)
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (16, 21, 30), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    border = (80, 220, 255) if not recovering_tracking else (60, 170, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), border, 2)

    tracking = "  |  HOLD - TRACKING" if recovering_tracking else ""
    cv2.putText(
        frame,
        title + tracking,
        (left + 12, top + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        border,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        hint,
        (left + 12, top + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (230, 234, 240),
        1,
        cv2.LINE_AA,
    )

    if mode == "transform" and transform_info is not None:
        scale, tilt_x, tilt_y, spin = transform_info
        values = f"{scale:.2f}x   X {tilt_x:+.0f}   Y {tilt_y:+.0f}   Z {spin:+.0f}"
        size = cv2.getTextSize(values, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        cv2.putText(
            frame,
            values,
            (right - size[0] - 12, top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_transform_mode_hud(frame: np.ndarray, control: str) -> None:
    """Keep the selected spatial control visible between manipulation gestures."""

    label = TRANSFORM_MODE_LABELS[control]
    height, width = frame.shape[:2]
    if height < 130 or width < 240:
        return
    shortcuts = "0 AUTO 1 SIZE 2 X 3 Y 4 Z" if width >= 560 else "[0-4]"
    text = f"3D {label} | {shortcuts}"
    font_scale = 0.38
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
    cv2.rectangle(
        frame,
        (8, height - 64),
        (min(width - 8, text_width + 24), height - 37),
        (18, 22, 30),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (15, height - 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )
