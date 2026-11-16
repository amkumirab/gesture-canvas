"""Dwell-operated 3D control menu with safe return to canvas gestures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .interaction_hud import TRANSFORM_MODE_LABELS
from .toolbar import DwellSelector, ToolButton


@dataclass(frozen=True, slots=True)
class MenuHand:
    key: str
    cursor: tuple[int, int]
    pointing: bool = False
    pinching: bool = False
    erasing: bool = False


class TransformControlMenu:
    """Pause canvas input during menu use and until a neutral hand is seen."""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.supported = width >= 240 and height >= 220
        right = min(width - 8, 318)
        self.trigger_rect = (8, height - 64, right, height - 37)
        panel_height = min(144, height - 152)
        self.panel_rect = (8, height - 70 - panel_height, min(width - 8, 428), height - 70)
        left, top, panel_right, bottom = self.panel_rect
        header = 24 if panel_height < 120 else 32
        row_height = (bottom - top - header - 12) // 2
        column_width = (panel_right - left - 16) // 3
        self.option_rects = {}
        self.close_rect = (0, 0, 0, 0)
        for index, mode in enumerate(("auto", "scale", "tilt_x", "tilt_y", "spin", "close")):
            row, column = divmod(index, 3)
            x = left + 5 + column * (column_width + 3)
            y = top + header + row * (row_height + 3)
            rect = (x, y, x + column_width, y + row_height)
            if mode == "close":
                self.close_rect = rect
            else:
                self.option_rects[mode] = rect
        self.is_open = False
        self._await_neutral = False
        self._over_trigger = False
        self._selector = DwellSelector(dwell_seconds=0.7)
        self._owner: str | None = None
        self._cursor: tuple[int, int] | None = None
        self._hover: str | None = None
        self._progress = 0.0

    @property
    def blocks_canvas(self) -> bool:
        return self.is_open or self._await_neutral or self._over_trigger

    def dismiss(self) -> None:
        if self.is_open:
            self._await_neutral = True
        self.is_open = False
        self._reset_hover()

    def update(
        self,
        hands: Sequence[MenuHand],
        now: float,
        enabled: bool = True,
    ) -> str | None:
        """Return a chosen mode once, retaining a single hand's dwell ownership."""

        self._over_trigger = False
        if not enabled or not self.supported:
            self.dismiss()
            return None
        neutral = bool(hands) and all(
            not hand.pointing and not hand.pinching and not hand.erasing for hand in hands
        )
        if neutral:
            self._await_neutral = False
        self._over_trigger = any(_inside(hand.cursor, self.trigger_rect) for hand in hands)
        if self._await_neutral:
            self._reset_hover()
            return None
        pointers = [
            hand for hand in hands
            if hand.pointing and not hand.pinching and self._target(hand.cursor) is not None
        ]
        pointer = next((hand for hand in pointers if hand.key == self._owner), None)
        if pointer is None and pointers:
            pointer = pointers[0]
        if pointer is None:
            self._reset_hover()
            return None
        if self._owner != pointer.key:
            self._reset_hover()
            self._owner = pointer.key
        self._cursor = pointer.cursor
        target = self._target(pointer.cursor)
        self._hover = target.key if target else None
        activated, self._progress = self._selector.update(target, now)
        if activated is None:
            return None
        if activated.key == "trigger" and not self.is_open:
            self.is_open = True
            return None
        mode = activated.key if activated.key in self.option_rects else None
        self.dismiss()
        return mode

    def draw(self, frame: np.ndarray, selected_mode: str) -> None:
        if not self.supported:
            return
        if self.is_open:
            left, top, right, bottom = self.panel_rect
            cv2.rectangle(frame, (left, top), (right, bottom), (18, 22, 30), -1)
            cv2.rectangle(frame, (left, top), (right, bottom), (80, 88, 100), 1)
            cv2.putText(
                frame, "3D CONTROL - POINT + HOLD", (left + 8, top + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 234, 240), 1, cv2.LINE_AA,
            )
            for mode, rect in self.option_rects.items():
                label = TRANSFORM_MODE_LABELS[mode]
                if mode == selected_mode:
                    label += " SELECTED"
                self._draw_button(frame, rect, mode, label, mode == selected_mode)
            self._draw_button(frame, self.close_rect, "close", "CLOSE", False)
        label = f"3D CONTROL: {TRANSFORM_MODE_LABELS[selected_mode]}"
        self._draw_button(frame, self.trigger_rect, "trigger", label + "  HOLD", self.is_open)
        if self._await_neutral:
            cv2.putText(
                frame, "LOWER POINTING FINGER TO RESUME", (10, self.height - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 220, 255), 1, cv2.LINE_AA,
            )
        if self._cursor is not None and self._hover is not None:
            cv2.circle(frame, self._cursor, 8, (80, 220, 255), 2, cv2.LINE_AA)

    def _target(self, point: tuple[int, int]) -> ToolButton | None:
        if _inside(point, self.trigger_rect):
            return ToolButton("trigger", "3D CONTROL")
        if self.is_open:
            for mode, rect in self.option_rects.items():
                if _inside(point, rect):
                    return ToolButton(mode, TRANSFORM_MODE_LABELS[mode])
            if _inside(point, self.close_rect):
                return ToolButton("close", "CLOSE")
        return None

    def _draw_button(self, frame, rect, key, label, selected) -> None:
        left, top, right, bottom = rect
        border = (80, 220, 255) if selected or key == self._hover else (80, 88, 100)
        cv2.rectangle(frame, (left, top), (right, bottom), (30, 36, 48), -1)
        cv2.rectangle(frame, (left, top), (right, bottom), border, 2 if selected else 1)
        font_scale = 0.4
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        if text_width > right - left - 10:
            font_scale *= (right - left - 10) / max(text_width, 1)
        cv2.putText(
            frame, label, (left + 5, top + (bottom - top) // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 247, 250), 1, cv2.LINE_AA,
        )
        if key == self._hover and self._progress > 0:
            end = left + 2 + round((right - left - 4) * self._progress)
            cv2.rectangle(frame, (left + 2, bottom - 4), (end, bottom - 2), border, -1)

    def _reset_hover(self) -> None:
        self._selector.reset()
        self._owner = None
        self._cursor = None
        self._hover = None
        self._progress = 0.0


def _inside(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom
