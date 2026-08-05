"""Drawing state with mask-based compositing and bounded undo history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class CanvasSnapshot:
    strokes: np.ndarray
    mask: np.ndarray


class DrawingCanvas:
    def __init__(self, width: int, height: int, history_limit: int = 30) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.width = width
        self.height = height
        self.history_limit = history_limit
        self.strokes = np.zeros((height, width, 3), dtype=np.uint8)
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self._undo: list[CanvasSnapshot] = []
        self._redo: list[CanvasSnapshot] = []
        self._stroke_active = False
        self._last_point: tuple[int, int] | None = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def begin_stroke(self, point: tuple[int, int]) -> None:
        if not self._stroke_active:
            self._push_undo()
            self._stroke_active = True
            self._last_point = point

    def add_point(
        self,
        point: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if thickness < 1:
            raise ValueError("thickness must be positive")
        if not self._stroke_active:
            self.begin_stroke(point)
        previous = self._last_point or point
        cv2.line(self.strokes, previous, point, color, thickness, cv2.LINE_AA)
        cv2.line(self.mask, previous, point, 255, thickness, cv2.LINE_AA)
        self._last_point = point

    def erase_point(self, point: tuple[int, int], thickness: int) -> None:
        if thickness < 1:
            raise ValueError("thickness must be positive")
        if not self._stroke_active:
            self.begin_stroke(point)
        previous = self._last_point or point
        cv2.line(self.strokes, previous, point, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.line(self.mask, previous, point, 0, thickness, cv2.LINE_AA)
        self._last_point = point

    def end_stroke(self) -> None:
        self._stroke_active = False
        self._last_point = None

    def clear(self) -> None:
        self.end_stroke()
        if not np.any(self.mask):
            return
        self._push_undo()
        self.strokes.fill(0)
        self.mask.fill(0)

    def undo(self) -> bool:
        self.end_stroke()
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        self.end_stroke()
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def composite(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Frame and canvas dimensions do not match")
        result = frame.copy()
        alpha = self.mask.astype(np.float32)[:, :, None] / 255.0
        blended = result.astype(np.float32) * (1.0 - alpha) + self.strokes * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def export(self, path: Path) -> Path:
        """Save strokes on a white background, independent of the webcam."""

        path.parent.mkdir(parents=True, exist_ok=True)
        white = np.full_like(self.strokes, 255)
        alpha = self.mask.astype(np.float32)[:, :, None] / 255.0
        image = white.astype(np.float32) * (1.0 - alpha) + self.strokes * alpha
        if not cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8)):
            raise OSError(f"Could not save drawing to {path}")
        return path

    def _snapshot(self) -> CanvasSnapshot:
        return CanvasSnapshot(self.strokes.copy(), self.mask.copy())

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        if len(self._undo) > self.history_limit:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, snapshot: CanvasSnapshot) -> None:
        self.strokes = snapshot.strokes.copy()
        self.mask = snapshot.mask.copy()

