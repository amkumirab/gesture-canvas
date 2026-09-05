"""Drawing state with mask-based compositing and bounded undo history."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from pathlib import Path

import cv2
import numpy as np

from .spatial import SpatialGuide, SpatialScene, SpatialSnapshot


@dataclass(slots=True)
class CanvasSnapshot:
    strokes: np.ndarray
    mask: np.ndarray
    spatial: SpatialSnapshot


@dataclass(slots=True)
class CanvasProjectState:
    width: int
    height: int
    history_limit: int
    current: CanvasSnapshot
    undo: list[CanvasSnapshot]
    redo: list[CanvasSnapshot]


@dataclass(slots=True)
class StrokeState:
    points: list[tuple[int, int]]
    color: tuple[int, int, int]
    thickness: int
    path_length: float = 0.0


@dataclass(slots=True)
class TransformState:
    start_midpoint: tuple[float, float]
    start_distance: float
    start_angle: float
    base_offset: tuple[int, int]
    base_scale: float
    base_rotation: float


@dataclass(slots=True)
class MoveState:
    original: CanvasSnapshot
    base: CanvasSnapshot
    selected_strokes: np.ndarray
    selected_mask: np.ndarray
    anchor: tuple[int, int]
    bounds: tuple[int, int, int, int]
    offset: tuple[int, int] = (0, 0)
    anchor_offset: tuple[int, int] = (0, 0)
    scale: float = 1.0
    rotation: float = 0.0
    rendered_bounds: tuple[int, int, int, int] | None = None
    transform: TransformState | None = None
    history_recorded: bool = False


class DrawingCanvas:
    def __init__(
        self,
        width: int,
        height: int,
        history_limit: int = 30,
        depth_sensitivity: float = 1.35,
    ) -> None:
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
        self._paint_stroke: StrokeState | None = None
        self._move: MoveState | None = None
        self._revision = 0
        self.spatial = SpatialScene(width, height, depth_sensitivity)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def revision(self) -> int:
        """Monotonically increasing identifier for rendered canvas changes."""

        return self._revision

    @property
    def is_moving(self) -> bool:
        return self._move is not None or self.spatial.is_moving

    @property
    def move_bounds(self) -> tuple[int, int, int, int] | None:
        if self._move is not None:
            return self._move.rendered_bounds
        return self.spatial.selected_bounds

    @property
    def transform_info(self) -> tuple[float, float] | None:
        if self._move is not None:
            return self._move.scale, self._move.rotation
        spatial_info = self.spatial.transform_info
        if spatial_info is None:
            return None
        scale, _, _, rotation_z = spatial_info
        return scale, rotation_z

    @property
    def spatial_transform_info(self) -> tuple[float, float, float, float] | None:
        return self.spatial.transform_info

    @property
    def spatial_transform_control(self) -> str | None:
        return self.spatial.transform_control

    @property
    def is_transforming(self) -> bool:
        return (
            self._move is not None and self._move.transform is not None
        ) or self.spatial.is_transforming

    @property
    def spatial_object_count(self) -> int:
        return self.spatial.object_count

    @property
    def selected_is_spatial(self) -> bool:
        return self._move is None and self.spatial.selected_index is not None

    @property
    def selected_spatial_z(self) -> float | None:
        return self.spatial.selected_z

    @property
    def spatial_guide(self) -> SpatialGuide | None:
        return self.spatial.selected_guide()

    def begin_stroke(self, point: tuple[int, int]) -> None:
        self.end_move()
        if not self._stroke_active:
            self._push_undo()
            self._stroke_active = True
            self._last_point = point
            self._paint_stroke = None

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
        if self._paint_stroke is None:
            self._paint_stroke = StrokeState([previous], color, thickness)
        if previous != point:
            self._paint_stroke.path_length += hypot(
                point[0] - previous[0], point[1] - previous[1]
            )
        self._paint_stroke.points.append(point)
        cv2.line(self.strokes, previous, point, color, thickness, cv2.LINE_AA)
        cv2.line(self.mask, previous, point, 255, thickness, cv2.LINE_AA)
        self._last_point = point
        self._touch()

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
        self._paint_stroke = None
        previous = self._continuous_previous(point, max_segment_length)
        cv2.line(self.strokes, previous, point, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.line(self.mask, previous, point, 0, thickness, cv2.LINE_AA)
        self._last_point = point
        self._touch()

    def end_stroke(self) -> None:
        if self._stroke_active and self._paint_stroke is not None:
            if self._fill_closed_stroke(self._paint_stroke):
                self._touch()
        self._stroke_active = False
        self._last_point = None
        self._paint_stroke = None

    def begin_move(
        self,
        point: tuple[int, int],
        selection_radius: int = 18,
        depth_signal: float = 0.0,
    ) -> bool:
        """Select the connected painted component at or near ``point``."""

        if selection_radius < 0:
            raise ValueError("selection_radius cannot be negative")
        self.end_stroke()
        self.end_move()
        if self.spatial.begin_move(point, depth_signal):
            self._touch()
            return True
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
            base=CanvasSnapshot(base_strokes, base_mask, self.spatial.snapshot()),
            selected_strokes=selected_strokes.astype(np.uint8),
            selected_mask=selected_mask,
            anchor=point,
            bounds=bounds,
            rendered_bounds=bounds,
        )
        self._render_move()
        return True

    def update_move(
        self,
        point: tuple[int, int],
        depth_signal: float = 0.0,
    ) -> bool:
        """Move the selected component relative to its initial grab point."""

        if self.spatial.is_moving:
            before = None if self.spatial.history_recorded else self._snapshot()
            changed = self.spatial.update_move(point, depth_signal)
            if changed and before is not None:
                self._record_undo(before)
                self.spatial.mark_history_recorded()
            if changed:
                self._touch()
            return changed
        if self._move is None:
            return False
        if self._move.transform is not None:
            return False
        dx = int(self._move.anchor_offset[0] + point[0] - self._move.anchor[0])
        dy = int(self._move.anchor_offset[1] + point[1] - self._move.anchor[1])
        if (dx, dy) == self._move.offset:
            return False
        self._ensure_move_history()
        self._move.offset = dx, dy
        self._render_move()
        self._touch()
        return True

    def begin_transform(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        minimum_distance: float = 24.0,
        first_depth_signal: float = 0.0,
        second_depth_signal: float = 0.0,
    ) -> bool:
        """Start a two-point scale and rotation gesture for the selected component."""

        if minimum_distance <= 0:
            raise ValueError("minimum_distance must be positive")
        if self.spatial.is_moving:
            return self.spatial.begin_transform(
                first,
                second,
                minimum_distance,
                first_depth_signal,
                second_depth_signal,
            )
        if self._move is None:
            return False
        distance = hypot(second[0] - first[0], second[1] - first[1])
        if distance < minimum_distance:
            return False
        midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        angle = atan2(second[1] - first[1], second[0] - first[0])
        self._move.transform = TransformState(
            start_midpoint=midpoint,
            start_distance=distance,
            start_angle=angle,
            base_offset=self._move.offset,
            base_scale=self._move.scale,
            base_rotation=self._move.rotation,
        )
        return True

    def update_transform(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        minimum_scale: float = 0.35,
        maximum_scale: float = 3.0,
        first_depth_signal: float = 0.0,
        second_depth_signal: float = 0.0,
    ) -> bool:
        """Update translation, scale, and rotation from two active pinch points."""

        if not 0 < minimum_scale <= maximum_scale:
            raise ValueError("Expected 0 < minimum_scale <= maximum_scale")
        if self.spatial.is_moving:
            before = None if self.spatial.history_recorded else self._snapshot()
            changed = self.spatial.update_transform(
                first,
                second,
                minimum_scale,
                maximum_scale,
                first_depth_signal,
                second_depth_signal,
            )
            if changed and before is not None:
                self._record_undo(before)
                self.spatial.mark_history_recorded()
            if changed:
                self._touch()
            return changed
        if self._move is None or self._move.transform is None:
            return False
        transform = self._move.transform
        distance = hypot(second[0] - first[0], second[1] - first[1])
        scale = float(
            np.clip(
                transform.base_scale * distance / transform.start_distance,
                minimum_scale,
                maximum_scale,
            )
        )
        current_angle = atan2(second[1] - first[1], second[0] - first[0])
        rotation = transform.base_rotation + degrees(
            current_angle - transform.start_angle
        )
        rotation = (rotation + 180.0) % 360.0 - 180.0
        midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        dx = int(round(transform.base_offset[0] + midpoint[0] - transform.start_midpoint[0]))
        dy = int(round(transform.base_offset[1] + midpoint[1] - transform.start_midpoint[1]))
        unchanged = (
            (dx, dy) == self._move.offset
            and abs(scale - self._move.scale) < 1e-4
            and abs(rotation - self._move.rotation) < 1e-3
        )
        if unchanged:
            return False
        self._ensure_move_history()
        self._move.offset = dx, dy
        self._move.scale = scale
        self._move.rotation = rotation
        self._render_move()
        self._touch()
        return True

    def end_transform(
        self,
        remaining_anchor: tuple[int, int] | None = None,
        remaining_depth_signal: float = 0.0,
    ) -> bool:
        """Leave two-hand mode while optionally continuing with one-hand movement."""

        if self.spatial.is_transforming:
            return self.spatial.end_transform(
                remaining_anchor,
                remaining_depth_signal,
            )
        if self._move is None or self._move.transform is None:
            return False
        self._move.transform = None
        if remaining_anchor is not None:
            self._move.anchor = remaining_anchor
            self._move.anchor_offset = self._move.offset
        return True

    def end_move(self) -> bool:
        """Finish a move operation and report whether anything changed."""

        if self.spatial.is_moving:
            return self.spatial.end_move()
        if self._move is None:
            return False
        changed = self._move.history_recorded
        self._move = None
        return changed

    def clear(self) -> None:
        self.end_stroke()
        self.end_move()
        if not np.any(self.mask) and self.spatial.object_count == 0:
            return
        self._push_undo()
        self.strokes.fill(0)
        self.mask.fill(0)
        self.spatial.clear()
        self._touch()

    def promote_to_3d(
        self,
        point: tuple[int, int],
        selection_radius: int = 18,
    ) -> bool:
        """Move a closed painted component into the perspective 3D workspace."""

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
        component_mask = (component.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return False
        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        fill_ratio = float(np.count_nonzero(component)) / max(width * height, 1)
        if (
            width < 16
            or height < 16
            or abs(cv2.contourArea(contour)) < 250
            or fill_ratio < 0.28
        ):
            return False

        pixels = self.strokes[component]
        color = tuple(int(value) for value in np.median(pixels, axis=0))
        original = self._snapshot()
        if not self.spatial.add_plane(contour, color):
            return False
        self._record_undo(original)
        self.strokes[component] = 0
        self.mask[component] = 0
        self._touch()
        return True

    def adjust_spatial_z(self, change: float) -> bool:
        """Move the selected object along Z as one undoable operation."""

        self.end_stroke()
        self.end_move()
        original = self._snapshot()
        if not self.spatial.adjust_z(change):
            return False
        self._record_undo(original)
        self._touch()
        return True

    def reset_spatial_rotation(self) -> bool:
        """Reset the selected spatial layer rotation as one undoable edit."""

        self.end_stroke()
        self.end_move()
        original = self._snapshot()
        if not self.spatial.reset_selected_rotation():
            return False
        self._record_undo(original)
        self._touch()
        return True

    def undo(self) -> bool:
        self.end_stroke()
        self.end_move()
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self._touch()
        return True

    def redo(self) -> bool:
        self.end_stroke()
        self.end_move()
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self._touch()
        return True

    def composite(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Frame and canvas dimensions do not match")
        result = frame.copy()
        alpha = self.mask.astype(np.float32)[:, :, None] / 255.0
        blended = result.astype(np.float32) * (1.0 - alpha) + self.strokes * alpha
        composited = np.clip(blended, 0, 255).astype(np.uint8)
        self.spatial.render(composited)
        return composited

    def export(self, path: Path) -> Path:
        """Save strokes on a white background, independent of the webcam."""

        path.parent.mkdir(parents=True, exist_ok=True)
        white = np.full_like(self.strokes, 255)
        alpha = self.mask.astype(np.float32)[:, :, None] / 255.0
        image = white.astype(np.float32) * (1.0 - alpha) + self.strokes * alpha
        rendered = np.clip(image, 0, 255).astype(np.uint8)
        self.spatial.render(rendered)
        if not cv2.imwrite(str(path), rendered):
            raise OSError(f"Could not save drawing to {path}")
        return path

    def project_state(
        self,
        finalize_interactions: bool = True,
        include_history: bool = True,
    ) -> CanvasProjectState:
        """Return detached editable state, optionally with undo/redo history."""

        if finalize_interactions:
            self.end_stroke()
            self.end_move()
        return CanvasProjectState(
            width=self.width,
            height=self.height,
            history_limit=self.history_limit,
            current=self._snapshot(),
            undo=(
                [self._copy_snapshot(snapshot) for snapshot in self._undo]
                if include_history
                else []
            ),
            redo=(
                [self._copy_snapshot(snapshot) for snapshot in self._redo]
                if include_history
                else []
            ),
        )

    def restore_project_state(self, state: CanvasProjectState) -> None:
        """Replace the canvas with a previously validated project state."""

        if (state.width, state.height) != (self.width, self.height):
            raise ValueError("Project state dimensions do not match the canvas")
        if state.history_limit < 1:
            raise ValueError("Project history limit must be positive")
        self.end_stroke()
        self.end_move()
        self.history_limit = state.history_limit
        self._restore(state.current)
        self._undo = [
            self._copy_snapshot(snapshot)
            for snapshot in state.undo[-self.history_limit :]
        ]
        self._redo = [
            self._copy_snapshot(snapshot)
            for snapshot in state.redo[-self.history_limit :]
        ]
        self._stroke_active = False
        self._last_point = None
        self._paint_stroke = None
        self._move = None
        self._touch()

    def _snapshot(self) -> CanvasSnapshot:
        return CanvasSnapshot(
            self.strokes.copy(),
            self.mask.copy(),
            self.spatial.snapshot(),
        )

    @staticmethod
    def _copy_snapshot(snapshot: CanvasSnapshot) -> CanvasSnapshot:
        return CanvasSnapshot(
            snapshot.strokes.copy(),
            snapshot.mask.copy(),
            SpatialSnapshot(
                objects=[shape.copy() for shape in snapshot.spatial.objects],
                selected_index=snapshot.spatial.selected_index,
            ),
        )

    def _touch(self) -> None:
        self._revision += 1

    def _push_undo(self) -> None:
        self._record_undo(self._snapshot())

    def _record_undo(self, snapshot: CanvasSnapshot) -> None:
        self._undo.append(snapshot)
        if len(self._undo) > self.history_limit:
            self._undo.pop(0)
        self._redo.clear()

    def _ensure_move_history(self) -> None:
        if self._move is not None and not self._move.history_recorded:
            self._record_undo(self._move.original)
            self._move.history_recorded = True

    def _restore(self, snapshot: CanvasSnapshot) -> None:
        self.strokes = snapshot.strokes.copy()
        self.mask = snapshot.mask.copy()
        self.spatial.restore(snapshot.spatial)

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

    def _fill_closed_stroke(self, stroke: StrokeState) -> bool:
        """Fill a sufficiently large closed stroke using its original color."""

        if len(stroke.points) < 6:
            return False
        points = np.asarray(stroke.points, dtype=np.int32)
        start = points[0]
        end = points[-1]
        closure_distance = hypot(float(end[0] - start[0]), float(end[1] - start[1]))
        closure_limit = max(18.0, stroke.thickness * 3.0)
        if closure_distance > closure_limit:
            return False

        x, y, width, height = cv2.boundingRect(points)
        minimum_dimension = max(16, stroke.thickness * 3)
        if width < minimum_dimension or height < minimum_dimension:
            return False
        if stroke.path_length < 1.15 * (width + height):
            return False

        contour = points.reshape((-1, 1, 2))
        minimum_area = float(minimum_dimension**2)
        if abs(cv2.contourArea(contour)) < minimum_area:
            return False

        cv2.fillPoly(self.strokes, [contour], stroke.color, lineType=cv2.LINE_AA)
        cv2.fillPoly(self.mask, [contour], 255, lineType=cv2.LINE_AA)
        return True

    def _render_move(self) -> None:
        if self._move is None:
            return
        x, y, width, height = self._move.bounds
        center = (x + (width - 1) / 2, y + (height - 1) / 2)
        transform = cv2.getRotationMatrix2D(
            center,
            self._move.rotation,
            self._move.scale,
        )
        transform[0, 2] += self._move.offset[0]
        transform[1, 2] += self._move.offset[1]
        size = (self.width, self.height)
        moved_strokes = cv2.warpAffine(
            self._move.selected_strokes,
            transform,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        moved_mask = cv2.warpAffine(
            self._move.selected_mask,
            transform,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        self.strokes = self._move.base.strokes.copy()
        self.mask = self._move.base.mask.copy()
        selected = moved_mask > 0
        self.strokes[selected] = moved_strokes[selected]
        self.mask = np.maximum(self.mask, moved_mask)
        ys, xs = np.nonzero(selected)
        if xs.size:
            self._move.rendered_bounds = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
        else:
            self._move.rendered_bounds = None

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
