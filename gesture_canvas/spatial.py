"""Lightweight extruded-shape scene rendered with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin

import cv2
import numpy as np


Color = tuple[int, int, int]


@dataclass(slots=True)
class ExtrudedShape:
    contour: np.ndarray
    color: Color
    position: tuple[float, float]
    depth: float
    scale: float = 1.0
    rotation_x: float = -18.0
    rotation_y: float = 24.0
    rotation_z: float = 0.0

    def copy(self) -> ExtrudedShape:
        return ExtrudedShape(
            contour=self.contour.copy(),
            color=self.color,
            position=self.position,
            depth=self.depth,
            scale=self.scale,
            rotation_x=self.rotation_x,
            rotation_y=self.rotation_y,
            rotation_z=self.rotation_z,
        )


@dataclass(slots=True)
class SpatialSnapshot:
    objects: list[ExtrudedShape]
    selected_index: int | None


@dataclass(slots=True)
class SpatialTransform:
    start_midpoint: tuple[float, float]
    start_distance: float
    start_angle: float
    base_scale: float
    base_rotation_x: float
    base_rotation_y: float
    base_rotation_z: float


@dataclass(slots=True)
class SpatialMove:
    object_index: int
    anchor: tuple[int, int]
    base_position: tuple[float, float]
    transform: SpatialTransform | None = None
    history_recorded: bool = False


class SpatialScene:
    """Own and manipulate simple meshes made by extruding 2D contours."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.objects: list[ExtrudedShape] = []
        self.selected_index: int | None = None
        self._move: SpatialMove | None = None

    @property
    def is_moving(self) -> bool:
        return self._move is not None

    @property
    def is_transforming(self) -> bool:
        return self._move is not None and self._move.transform is not None

    @property
    def history_recorded(self) -> bool:
        return self._move is not None and self._move.history_recorded

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def selected_bounds(self) -> tuple[int, int, int, int] | None:
        shape = self._selected()
        if shape is None:
            return None
        projected, _ = self._project(shape)
        x, y, width, height = cv2.boundingRect(projected.astype(np.float32))
        return x, y, width, height

    @property
    def transform_info(self) -> tuple[float, float, float, float] | None:
        shape = self._selected()
        if shape is None:
            return None
        return shape.scale, shape.rotation_x, shape.rotation_y, shape.rotation_z

    @property
    def selected_depth(self) -> float | None:
        shape = self._selected()
        return None if shape is None else shape.depth

    def snapshot(self) -> SpatialSnapshot:
        return SpatialSnapshot(
            objects=[shape.copy() for shape in self.objects],
            selected_index=self.selected_index,
        )

    def restore(self, snapshot: SpatialSnapshot) -> None:
        self.objects = [shape.copy() for shape in snapshot.objects]
        self.selected_index = snapshot.selected_index
        self._move = None

    def clear(self) -> None:
        self.objects.clear()
        self.selected_index = None
        self._move = None

    def add_extrusion(
        self,
        contour: np.ndarray,
        color: Color,
        depth: float | None = None,
    ) -> bool:
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            return False

        curve = points.reshape(-1, 1, 2)
        perimeter = cv2.arcLength(curve, True)
        simplified = cv2.approxPolyDP(curve, max(1.5, perimeter * 0.012), True)
        points = simplified.reshape(-1, 2)
        if len(points) < 3 or abs(cv2.contourArea(points)) < 250:
            return False

        x, y, width, height = cv2.boundingRect(points)
        center = (x + (width - 1) / 2, y + (height - 1) / 2)
        local = points - np.asarray(center, dtype=np.float32)
        extrusion_depth = (
            float(np.clip(min(width, height) * 0.45, 18, 120))
            if depth is None
            else float(depth)
        )
        if extrusion_depth <= 0:
            raise ValueError("depth must be positive")

        self.objects.append(
            ExtrudedShape(
                contour=local,
                color=color,
                position=center,
                depth=extrusion_depth,
            )
        )
        self.selected_index = len(self.objects) - 1
        self._move = None
        return True

    def begin_move(self, point: tuple[int, int]) -> bool:
        self.end_move()
        for index in range(len(self.objects) - 1, -1, -1):
            if self._hit_test(self.objects[index], point):
                self.selected_index = index
                self._move = SpatialMove(
                    object_index=index,
                    anchor=point,
                    base_position=self.objects[index].position,
                )
                return True
        return False

    def update_move(self, point: tuple[int, int]) -> bool:
        if self._move is None or self._move.transform is not None:
            return False
        shape = self.objects[self._move.object_index]
        position = (
            self._move.base_position[0] + point[0] - self._move.anchor[0],
            self._move.base_position[1] + point[1] - self._move.anchor[1],
        )
        if np.allclose(position, shape.position):
            return False
        shape.position = position
        return True

    def begin_transform(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        minimum_distance: float = 24.0,
    ) -> bool:
        if minimum_distance <= 0:
            raise ValueError("minimum_distance must be positive")
        if self._move is None:
            return False
        distance = hypot(second[0] - first[0], second[1] - first[1])
        if distance < minimum_distance:
            return False
        shape = self.objects[self._move.object_index]
        midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        self._move.transform = SpatialTransform(
            start_midpoint=midpoint,
            start_distance=distance,
            start_angle=atan2(second[1] - first[1], second[0] - first[0]),
            base_scale=shape.scale,
            base_rotation_x=shape.rotation_x,
            base_rotation_y=shape.rotation_y,
            base_rotation_z=shape.rotation_z,
        )
        return True

    def update_transform(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        minimum_scale: float = 0.35,
        maximum_scale: float = 3.0,
    ) -> bool:
        if not 0 < minimum_scale <= maximum_scale:
            raise ValueError("Expected 0 < minimum_scale <= maximum_scale")
        if self._move is None or self._move.transform is None:
            return False

        shape = self.objects[self._move.object_index]
        transform = self._move.transform
        distance = hypot(second[0] - first[0], second[1] - first[1])
        scale = float(
            np.clip(
                transform.base_scale * distance / transform.start_distance,
                minimum_scale,
                maximum_scale,
            )
        )
        angle = atan2(second[1] - first[1], second[0] - first[0])
        rotation_z = _normalize_angle(
            transform.base_rotation_z + degrees(angle - transform.start_angle)
        )
        midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        rotation_x = _normalize_angle(
            transform.base_rotation_x
            + (midpoint[1] - transform.start_midpoint[1]) * 0.65
        )
        rotation_y = _normalize_angle(
            transform.base_rotation_y
            + (midpoint[0] - transform.start_midpoint[0]) * 0.65
        )
        unchanged = (
            abs(scale - shape.scale) < 1e-4
            and abs(rotation_x - shape.rotation_x) < 1e-3
            and abs(rotation_y - shape.rotation_y) < 1e-3
            and abs(rotation_z - shape.rotation_z) < 1e-3
        )
        if unchanged:
            return False
        shape.scale = scale
        shape.rotation_x = rotation_x
        shape.rotation_y = rotation_y
        shape.rotation_z = rotation_z
        return True

    def end_transform(self, remaining_anchor: tuple[int, int] | None = None) -> bool:
        if self._move is None or self._move.transform is None:
            return False
        self._move.transform = None
        if remaining_anchor is not None:
            shape = self.objects[self._move.object_index]
            self._move.anchor = remaining_anchor
            self._move.base_position = shape.position
        return True

    def mark_history_recorded(self) -> None:
        if self._move is not None:
            self._move.history_recorded = True

    def end_move(self) -> bool:
        if self._move is None:
            return False
        changed = self._move.history_recorded
        self._move = None
        return changed

    def adjust_depth(self, change: float) -> bool:
        shape = self._selected()
        if shape is None:
            return False
        depth = float(np.clip(shape.depth + change, 8, 240))
        if abs(depth - shape.depth) < 1e-6:
            return False
        shape.depth = depth
        return True

    def render(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Frame and scene dimensions do not match")
        for shape in self.objects:
            self._render_shape(frame, shape)

    def _selected(self) -> ExtrudedShape | None:
        if self.selected_index is None or self.selected_index >= len(self.objects):
            return None
        return self.objects[self.selected_index]

    def _hit_test(self, shape: ExtrudedShape, point: tuple[int, int]) -> bool:
        projected, faces = self._project(shape)
        for indices, _, _ in faces:
            polygon = projected[indices].astype(np.float32)
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return True
        return False

    def _render_shape(self, frame: np.ndarray, shape: ExtrudedShape) -> None:
        projected, faces = self._project(shape)
        faces = sorted(faces, key=lambda face: face[1], reverse=True)
        for indices, _, shade in faces:
            polygon = np.rint(projected[indices]).astype(np.int32)
            cv2.fillPoly(frame, [polygon], _shade(shape.color, shade), cv2.LINE_AA)
            cv2.polylines(frame, [polygon], True, _shade(shape.color, 0.35), 1, cv2.LINE_AA)

    def _project(
        self,
        shape: ExtrudedShape,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, float, float]]]:
        count = len(shape.contour)
        front = np.column_stack(
            (shape.contour, np.full(count, -shape.depth / 2, dtype=np.float32))
        )
        back = np.column_stack(
            (shape.contour, np.full(count, shape.depth / 2, dtype=np.float32))
        )
        vertices = np.vstack((front, back)) * shape.scale
        rotated = vertices @ _rotation_matrix(
            shape.rotation_x,
            shape.rotation_y,
            shape.rotation_z,
        ).T

        focal_length = max(self.width, self.height) * 1.8
        perspective = focal_length / np.maximum(focal_length + rotated[:, 2], 1.0)
        projected = np.column_stack(
            (
                shape.position[0] + rotated[:, 0] * perspective,
                shape.position[1] + rotated[:, 1] * perspective,
            )
        ).astype(np.float32)

        faces: list[tuple[np.ndarray, float, float]] = [
            (
                np.arange(count, dtype=np.int32),
                float(rotated[:count, 2].mean()),
                1.0,
            ),
            (
                np.arange(count, count * 2, dtype=np.int32)[::-1],
                float(rotated[count:, 2].mean()),
                0.55,
            ),
        ]
        for index in range(count):
            following = (index + 1) % count
            indices = np.asarray(
                [index, following, following + count, index + count],
                dtype=np.int32,
            )
            shade = 0.58 + 0.09 * (index % 3)
            faces.append((indices, float(rotated[indices, 2].mean()), shade))
        return projected, faces


def _rotation_matrix(x_degrees: float, y_degrees: float, z_degrees: float) -> np.ndarray:
    x, y, z = map(radians, (x_degrees, y_degrees, z_degrees))
    rx = np.asarray(((1, 0, 0), (0, cos(x), -sin(x)), (0, sin(x), cos(x))))
    ry = np.asarray(((cos(y), 0, sin(y)), (0, 1, 0), (-sin(y), 0, cos(y))))
    rz = np.asarray(((cos(z), -sin(z), 0), (sin(z), cos(z), 0), (0, 0, 1)))
    return (rz @ ry @ rx).astype(np.float32)


def _normalize_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _shade(color: Color, factor: float) -> Color:
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)
