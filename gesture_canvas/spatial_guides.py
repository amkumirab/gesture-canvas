"""Visual orientation and depth guides for the selected spatial layer."""

from __future__ import annotations

from math import hypot

import cv2
import numpy as np

from .spatial import SpatialGuide


X_COLOR = (70, 80, 245)
Y_COLOR = (80, 220, 90)
Z_COLOR = (245, 165, 55)
PANEL_COLOR = (22, 27, 36)
TEXT_COLOR = (225, 230, 238)


def depth_marker_y(guide: SpatialGuide, top: int, bottom: int) -> int:
    """Map the selected Z coordinate onto a near-at-top depth ruler."""

    if bottom <= top:
        raise ValueError("bottom must be greater than top")
    span = guide.maximum_z - guide.minimum_z
    if span <= 0:
        raise ValueError("guide depth range must be positive")
    progress = (guide.maximum_z - guide.z_position) / span
    progress = float(np.clip(progress, 0.0, 1.0))
    return int(round(top + progress * (bottom - top)))


def draw_spatial_guides(
    frame: np.ndarray,
    guide: SpatialGuide | None,
    content_top: int = 74,
    active_control: str | None = None,
) -> None:
    """Draw projected XYZ axes and a responsive Z-position ruler in-place."""

    if guide is None:
        return
    height, width = frame.shape[:2]
    if height < 100 or width < 180:
        return

    active_axis = {
        "tilt_x": "X",
        "tilt_y": "Y",
        "spin": "Z",
    }.get(active_control)
    _draw_axis(
        frame,
        guide.origin,
        guide.x_axis,
        "X",
        X_COLOR,
        (7, -7),
        active_axis,
    )
    _draw_axis(
        frame,
        guide.origin,
        guide.y_axis,
        "Y",
        Y_COLOR,
        (7, 16),
        active_axis,
    )
    z_color = _axis_color(Z_COLOR, active_axis, "Z")
    z_thickness = 4 if active_axis == "Z" else 2
    if hypot(
        guide.z_axis[0] - guide.origin[0],
        guide.z_axis[1] - guide.origin[1],
    ) >= 6:
        _draw_axis(
            frame,
            guide.origin,
            guide.z_axis,
            "Z",
            Z_COLOR,
            (-15, -7),
            active_axis,
        )
    else:
        cv2.circle(
            frame,
            guide.origin,
            7,
            z_color,
            z_thickness,
            cv2.LINE_AA,
        )
        cv2.circle(frame, guide.origin, 2, z_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Z",
            (guide.origin[0] - 17, guide.origin[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            z_color,
            2 if active_axis == "Z" else 1,
            cv2.LINE_AA,
        )

    ruler_top = max(content_top + 28, 24)
    ruler_bottom = min(height - 72, ruler_top + 220)
    if ruler_bottom - ruler_top < 90:
        return

    panel_right = width - 10
    panel_left = max(8, panel_right - 108)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_left, ruler_top - 18),
        (panel_right, ruler_bottom + 36),
        PANEL_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(
        frame,
        (panel_left, ruler_top - 18),
        (panel_right, ruler_bottom + 36),
        (75, 84, 98),
        1,
    )

    ruler_x = panel_right - 20
    cv2.line(
        frame,
        (ruler_x, ruler_top),
        (ruler_x, ruler_bottom),
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    for index in range(5):
        tick_y = round(ruler_top + index * (ruler_bottom - ruler_top) / 4)
        cv2.line(frame, (ruler_x - 5, tick_y), (ruler_x + 5, tick_y), TEXT_COLOR, 1)

    marker_y = depth_marker_y(guide, ruler_top, ruler_bottom)
    marker = np.asarray(
        (
            (ruler_x - 2, marker_y),
            (ruler_x - 14, marker_y - 7),
            (ruler_x - 14, marker_y + 7),
        ),
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [marker], Z_COLOR, cv2.LINE_AA)
    cv2.putText(
        frame,
        "NEAR",
        (panel_left + 8, ruler_top + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "FAR",
        (panel_left + 8, ruler_bottom + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Z {guide.z_position:+.0f}",
        (panel_left + 8, ruler_bottom + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        Z_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_axis(
    frame: np.ndarray,
    origin: tuple[int, int],
    endpoint: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    label_offset: tuple[int, int],
    active_axis: str | None,
) -> None:
    axis_color = _axis_color(color, active_axis, label)
    thickness = 4 if active_axis == label else 2
    cv2.arrowedLine(
        frame,
        origin,
        endpoint,
        axis_color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.22,
    )
    cv2.putText(
        frame,
        label,
        (endpoint[0] + label_offset[0], endpoint[1] + label_offset[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        axis_color,
        2 if active_axis == label else 1,
        cv2.LINE_AA,
    )


def _axis_color(
    color: tuple[int, int, int],
    active_axis: str | None,
    axis: str,
) -> tuple[int, int, int]:
    if active_axis is None or active_axis == axis:
        return color
    return tuple(round(channel * 0.38) for channel in color)
