"""Drawing state with mask-based compositing and bounded undo history."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class CanvasSnapshot:
    strokes: np.ndarray
    mask: np.ndarray


@dataclass(slots=True)
class MoveState:
    original: CanvasSnapshot
    base: CanvasSnapshot
    selected_strokes: np.ndarray
    selected_mask: np.ndarray
    anchor: tuple[int, int]
    bounds: tuple[int, int, int, int]
    offset: tuple[int, int] = (0, 0)
    history_recorded: bool = False


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
        self._move: MoveState | None = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def is_moving(self) -> bool:
        return self._move is not None

    @property
    def move_bounds(self) -> tuple[int, int, int, int] | None:
        if self._move is None:
            return None
        x, y, width, height = self._move.bounds
        dx, dy = self._move.offset
        return x + dx, y + dy, width, height

    def begin_stroke(self, point: tuple[int, int]) -> None:
        self.end_move()
        if not self._stroke_active:
            self._push_undo()
            self._stroke_active = True
            self._last_point = point

    def add_point(
        self,
        point: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        max_segment_length: float | None = None,
    ) -> None:
        if thickness < 1:
            raise ValueError("thickness must be positive")
        if not self._stroke_active:
            self.begin_stroke(point)
        previous = self._continuous_previous(point, max_segment_length)
        cv2.line(self.strokes, previous, point, color, thickness, cv2.LINE_AA)
        cv2.line(self.mask, previous, point, 255, thickness, cv2.LINE_AA)
        self._last_point = point

    def erase_point(
        self,
        point: tuple[int, int],
        thickness: int,
        max_segment_length: float | None = None,
    ) -> None:
        if thickness < 1:
            raise ValueError("thickness must be positive")
        if not self._stroke_active:
            self.begin_stroke(point)
        previous = self._continuous_previous(point, max_segment_length)
        cv2.line(self.strokes, previous, point, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.line(self.mask, previous, point, 0, thickness, cv2.LINE_AA)
        self._last_point = point

    def end_stroke(self) -> None:
        self._stroke_active = False
        self._last_point = None

    def begin_move(self, point: tuple[int, int], selection_radius: int = 18) -> bool:
        """Select the connected painted component at or near ``point``."""

        if selection_radius < 0:
            raise ValueError("selection_radius cannot be negative")
        self.end_stroke()
        self.end_move()
        binary = (self.mask > 0).astype(np.uint8)
        if not np.any(binary):
            return False

        _, labels = cv2.connectedComponents(binary, connectivity=8)
        label = self._nearest_label(labels, point, selection_radius)
        if label == 0:
            return False

        component = labels == label
        ys, xs = np.nonzero(component)
        bounds = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
        original = self._snapshot()
        selected_strokes = np.where(component[:, :, None], self.strokes, 0)
        selected_mask = np.where(component, self.mask, 0).astype(np.uint8)
        base_strokes = self.strokes.copy()
        base_mask = self.mask.copy()
        base_strokes[component] = 0
        base_mask[component] = 0
        self._move = MoveState(
            original=original,
            base=CanvasSnapshot(base_strokes, base_mask),
            selected_strokes=selected_strokes.astype(np.uint8),
            selected_mask=selected_mask,
            anchor=point,
            bounds=bounds,
        )
        self._render_move(0, 0)
        return True

    def update_move(self, point: tuple[int, int]) -> bool:
        """Move the selected component relative to its initial grab point."""

        if self._move is None:
            return False
        dx = int(point[0] - self._move.anchor[0])
        dy = int(point[1] - self._move.anchor[1])
        if (dx, dy) == self._move.offset:
            return False
        if not self._move.history_recorded:
            self._record_undo(self._move.original)
            self._move.history_recorded = True
        self._move.offset = dx, dy
        self._render_move(dx, dy)
        return True

    def end_move(self) -> bool:
        """Finish a move operation and report whether anything changed."""

        if self._move is None:
            return False
        changed = self._move.history_recorded
        self._move = None
        return changed

    def clear(self) -> None:
        self.end_stroke()
        self.end_move()
        if not np.any(self.mask):
            return
        self._push_undo()
        self.strokes.fill(0)
        self.mask.fill(0)

    def undo(self) -> bool:
        self.end_stroke()
        self.end_move()
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        self.end_stroke()
        self.end_move()
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
        self._record_undo(self._snapshot())

    def _record_undo(self, snapshot: CanvasSnapshot) -> None:
        self._undo.append(snapshot)
        if len(self._undo) > self.history_limit:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, snapshot: CanvasSnapshot) -> None:
        self.strokes = snapshot.strokes.copy()
        self.mask = snapshot.mask.copy()

    def _continuous_previous(
        self,
        point: tuple[int, int],
        max_segment_length: float | None,
    ) -> tuple[int, int]:
        if max_segment_length is not None and max_segment_length <= 0:
            raise ValueError("max_segment_length must be positive")
        previous = self._last_point or point
        if max_segment_length is not None:
            distance = hypot(point[0] - previous[0], point[1] - previous[1])
            if distance > max_segment_length:
                return point
        return previous

    def _render_move(self, dx: int, dy: int) -> None:
        if self._move is None:
            return
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        size = (self.width, self.height)
        moved_strokes = cv2.warpAffine(
            self._move.selected_strokes,
            transform,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        moved_mask = cv2.warpAffine(
            self._move.selected_mask,
            transform,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        self.strokes = self._move.base.strokes.copy()
        self.mask = self._move.base.mask.copy()
        selected = moved_mask > 0
        self.strokes[selected] = moved_strokes[selected]
        self.mask = np.maximum(self.mask, moved_mask)

    @staticmethod
    def _nearest_label(
        labels: np.ndarray,
        point: tuple[int, int],
        radius: int,
    ) -> int:
        height, width = labels.shape
        x = int(np.clip(point[0], 0, width - 1))
        y = int(np.clip(point[1], 0, height - 1))
        if labels[y, x] != 0:
            return int(labels[y, x])
        if radius == 0:
            return 0

        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        nearby = labels[y0:y1, x0:x1]
        candidates = np.argwhere(nearby > 0)
        if candidates.size == 0:
            return 0
        global_y = candidates[:, 0] + y0
        global_x = candidates[:, 1] + x0
        distances = (global_x - x) ** 2 + (global_y - y) ** 2
        nearest = int(np.argmin(distances))
        if int(distances[nearest]) > radius**2:
            return 0
        return int(labels[global_y[nearest], global_x[nearest]])
