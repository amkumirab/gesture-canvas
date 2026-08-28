"""Clear on-screen feedback for direct manipulation gestures."""

from __future__ import annotations

import cv2
import numpy as np


def draw_manipulation_hud(
    frame: np.ndarray,
    mode: str,
    transform_info: tuple[float, float, float, float] | None = None,
    recovering_tracking: bool = False,
    content_top: int = 74,
    active_control: str | None = None,
) -> None:
    """Draw a compact gesture hint only while an object is being manipulated."""

    if mode == "idle" or frame.shape[0] < 130 or frame.shape[1] < 320:
        return
    if mode not in {"move", "move-3d", "transform"}:
        raise ValueError(f"Unknown manipulation mode: {mode}")

    if mode == "transform":
        title = "TWO-HAND TRANSFORM"
        control_labels = {
            "scale": "SCALE",
            "spin": "SPIN Z",
            "tilt_x": "TILT X",
            "tilt_y": "TILT Y",
        }
        if active_control in control_labels:
            hint = (
                f"{control_labels[active_control]} LOCKED   "
                "RELEASE SECOND PINCH TO SWITCH"
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
