"""In-app controls reference rendered over the camera view."""

from __future__ import annotations

import cv2
import numpy as np


HELP_ITEMS = (
    ("Calibrate", "K"),
    ("Restore defaults", "D"),
    ("Draw", "Raise index finger"),
    ("Erase", "Raise index and middle fingers"),
    ("Create 3D layer", "Point inside a closed shape, then E"),
    ("Select tool", "Point at toolbar and hold"),
    ("Move X / Y", "Pinch a drawing and drag"),
    ("Move Z", "While pinching, move hand closer / farther"),
    ("Scale", "Two pinches: move hands apart / together"),
    ("Spin", "Two pinches: turn the line between hands"),
    ("Tilt X / Y", "Move both hands together or change hand depth"),
    ("Switch transform", "Release second pinch, then pinch again"),
    ("Reset rotation", "R"),
    ("Tracking recovery", "Keep pinching through brief hand loss"),
    ("3D guides", "Shown automatically for selected layer"),
    ("Move Z by key", "- farther    + closer"),
    ("Brush size", "[ thinner    ] thicker"),
    ("Undo / redo", "Z / Y"),
    ("Save / clear", "S / C"),
    ("Close help", "H"),
)


def draw_help_overlay(frame: np.ndarray, content_top: int = 74) -> None:
    """Draw a responsive, translucent controls panel in-place."""

    height, width = frame.shape[:2]
    if height < 100 or width < 180:
        return

    margin = max(8, min(24, width // 30))
    left = margin
    right = width - margin
    top = min(max(margin, content_top + 12), height - 72)
    bottom = height - margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (15, 19, 27), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), (80, 220, 255), 2)

    cv2.putText(
        frame,
        "CONTROLS  [H] CLOSE",
        (left + 18, top + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72 if width >= 500 else 0.46,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )

    available_height = max(0, bottom - top - 54)
    line_height = max(23, min(28, available_height // max(len(HELP_ITEMS), 1)))
    available_lines = max(0, (bottom - top - 54) // line_height)
    value_x = left + min(180, max(105, (right - left) // 3))
    font_scale = 0.5 if width >= 500 else 0.4
    for index, (action, control) in enumerate(HELP_ITEMS[:available_lines]):
        y = top + 62 + index * line_height
        cv2.putText(
            frame,
            action,
            (left + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            control,
            (value_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (190, 198, 210),
            1,
            cv2.LINE_AA,
        )
