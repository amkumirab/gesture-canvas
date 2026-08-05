"""OpenCV toolbar rendering and hit testing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ToolButton:
    key: str
    label: str
    color: tuple[int, int, int] | None = None


class DwellSelector:
    """Activate a toolbar button after the pointer rests on it.

    Dwell selection is intentionally independent of a precise pinch threshold,
    which varies considerably between users and camera angles.
    """

    def __init__(self, dwell_seconds: float = 0.7) -> None:
        if dwell_seconds <= 0:
            raise ValueError("dwell_seconds must be positive")
        self.dwell_seconds = dwell_seconds
        self._button_key: str | None = None
        self._started_at = 0.0
        self._fired = False

    def update(
        self,
        button: ToolButton | None,
        now: float,
    ) -> tuple[ToolButton | None, float]:
        """Return the activated button and current progress from zero to one."""

        if button is None:
            self.reset()
            return None, 0.0

        if button.key != self._button_key:
            self._button_key = button.key
            self._started_at = now
            self._fired = False
            return None, 0.0

        elapsed = max(0.0, now - self._started_at)
        ready = elapsed + 1e-9 >= self.dwell_seconds
        progress = 1.0 if ready else elapsed / self.dwell_seconds
        if ready and not self._fired:
            self._fired = True
            return button, progress
        return None, progress

    def reset(self) -> None:
        self._button_key = None
        self._started_at = 0.0
        self._fired = False


BUTTONS = (
    ToolButton("blue", "BLUE", (255, 90, 30)),
    ToolButton("green", "GREEN", (60, 210, 70)),
    ToolButton("red", "RED", (50, 60, 240)),
    ToolButton("black", "BLACK", (20, 20, 20)),
    ToolButton("eraser", "ERASE"),
    ToolButton("undo", "UNDO"),
    ToolButton("redo", "REDO"),
    ToolButton("clear", "CLEAR"),
    ToolButton("save", "SAVE"),
)


class Toolbar:
    height = 74

    def __init__(self, frame_width: int) -> None:
        self.frame_width = frame_width
        self.margin = 8
        self.gap = 5
        usable = frame_width - 2 * self.margin - self.gap * (len(BUTTONS) - 1)
        self.button_width = max(54, usable // len(BUTTONS))

    def hit_test(self, point: tuple[int, int]) -> ToolButton | None:
        x, y = point
        if y < self.margin or y > self.height - self.margin:
            return None
        for index, button in enumerate(BUTTONS):
            left = self.margin + index * (self.button_width + self.gap)
            if left <= x <= left + self.button_width:
                return button
        return None

    def draw(
        self,
        frame: np.ndarray,
        active_key: str,
        hover_key: str | None = None,
        hover_progress: float = 0.0,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.height), (18, 22, 30), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        for index, button in enumerate(BUTTONS):
            left = self.margin + index * (self.button_width + self.gap)
            right = min(left + self.button_width, self.frame_width - self.margin)
            selected = button.key == active_key
            hovered = button.key == hover_key
            border = (80, 220, 255) if selected or hovered else (80, 88, 100)
            fill = button.color if button.color is not None else (42, 48, 60)
            cv2.rectangle(frame, (left, self.margin), (right, self.height - self.margin), fill, -1)
            cv2.rectangle(frame, (left, self.margin), (right, self.height - self.margin), border, 2)

            if hovered and hover_progress > 0:
                progress_right = left + round((right - left) * min(1.0, hover_progress))
                cv2.rectangle(
                    frame,
                    (left + 2, self.height - self.margin - 5),
                    (progress_right, self.height - self.margin - 2),
                    (80, 220, 255),
                    -1,
                )

            text_color = (255, 255, 255)
            if button.key in {"green", "red"}:
                text_color = (20, 20, 20)
            font_scale = 0.42 if self.button_width < 80 else 0.5
            size = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            tx = left + max(3, (right - left - size[0]) // 2)
            ty = self.margin + (self.height - 2 * self.margin + size[1]) // 2
            cv2.putText(frame, button.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
