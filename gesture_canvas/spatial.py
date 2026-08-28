"""Perspective scene for flat drawings placed in three-dimensional space."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, log, radians, sin

import cv2
import numpy as np


Color = tuple[int, int, int]


@dataclass(slots=True)
class SpatialShape:
    contour: np.ndarray
    color: Color
    position: tuple[float, float, float]
    scale: float = 1.0
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0

    def copy(self) -> SpatialShape:
        return SpatialShape(
            contour=self.contour.copy(),
            color=self.color,
            position=self.position,
            scale=self.scale,
            rotation_x=self.rotation_x,
            rotation_y=self.rotation_y,
            rotation_z=self.rotation_z,
        )


@dataclass(slots=True)
class SpatialSnapshot:
    objects: list[SpatialShape]
    selected_index: int | None


@dataclass(frozen=True, slots=True)
class SpatialGuide:
    origin: tuple[int, int]
    x_axis: tuple[int, int]
    y_axis: tuple[int, int]
    z_axis: tuple[int, int]
    z_position: float
    minimum_z: float
    maximum_z: float


@dataclass(slots=True)
class SpatialTransform:
    start_midpoint: tuple[float, float]
    start_distance: float
    start_angle: float
    base_scale: float
    base_rotation_x: float
    base_rotation_y: float
    base_rotation_z: float
    start_depth_balance: float | None


@dataclass(slots=True)
class SpatialMove:
    object_index: int
    anchor: tuple[int, int]
    base_position: tuple[float, float, float]
    start_depth_signal: float | None
    transform: SpatialTransform | None = None
    history_recorded: bool = False


class SpatialScene:
    """Own and manipulate flat drawings in a perspective 3D workspace."""

    depth_dead_zone = 0.025
    depth_sensitivity = 1.35

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.objects: list[SpatialShape] = []
        self.selected_index: int | None = None
        self._move: SpatialMove | None = None

    @property
    def focal_length(self) -> float:
        return max(self.width, self.height) * 1.8

    @property
    def z_limits(self) -> tuple[float, float]:
        return -self.focal_length * 2.0, self.focal_length * 0.6

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
        return cv2.boundingRect(projected.astype(np.float32))

    @property
    def transform_info(self) -> tuple[float, float, float, float] | None:
        shape = self._selected()
        if shape is None:
            return None
        return shape.scale, shape.rotation_x, shape.rotation_y, shape.rotation_z

    @property
    def selected_z(self) -> float | None:
        shape = self._selected()
        return None if shape is None else shape.position[2]

    def selected_guide(self, axis_length: float = 54.0) -> SpatialGuide | None:
        """Return projected local axes and depth range for the selected shape."""

        if axis_length <= 0:
            raise ValueError("axis_length must be positive")
        shape = self._selected()
        if shape is None:
            return None
        local_length = axis_length / max(shape.scale, 0.01)
        axes = np.asarray(
            (
                (0, 0, 0),
                (local_length, 0, 0),
                (0, local_length, 0),
                (0, 0, local_length),
            ),
            dtype=np.float32,
        )
        projected = np.rint(self._project_local_points(shape, axes)).astype(int)
        minimum_z, maximum_z = self.z_limits
        return SpatialGuide(
            origin=tuple(projected[0]),
            x_axis=tuple(projected[1]),
            y_axis=tuple(projected[2]),
            z_axis=tuple(projected[3]),
            z_position=shape.position[2],
            minimum_z=minimum_z,
            maximum_z=maximum_z,
        )

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

    def add_plane(
        self,
        contour: np.ndarray,
        color: Color,
        z_position: float = 0.0,
    ) -> bool:
        """Add a simplified filled contour as a flat object in 3D space."""

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
        z_position = self._clamp_z(float(z_position))
        self.objects.append(
            SpatialShape(
                contour=local,
                color=color,
                position=(center[0], center[1], z_position),
            )
        )
        self.selected_index = len(self.objects) - 1
        self._move = None
        return True

    def begin_move(
        self,
        point: tuple[int, int],
        depth_signal: float = 0.0,
    ) -> bool:
        self.end_move()
        indices = sorted(
            range(len(self.objects)),
            key=lambda index: self.objects[index].position[2],
            reverse=True,
        )
        for index in indices:
            if self._hit_test(self.objects[index], point):
                self.selected_index = index
                self._move = SpatialMove(
                    object_index=index,
                    anchor=point,
                    base_position=self.objects[index].position,
                    start_depth_signal=(
                        float(depth_signal) if depth_signal > 1e-6 else None
                    ),
                )
                return True
        return False

    def update_move(
        self,
        point: tuple[int, int],
        depth_signal: float = 0.0,
    ) -> bool:
        if self._move is None or self._move.transform is not None:
            return False
        shape = self.objects[self._move.object_index]
        z_position = self._depth_from_signal(depth_signal)
        position = (
            self._move.base_position[0] + point[0] - self._move.anchor[0],
            self._move.base_position[1] + point[1] - self._move.anchor[1],
            z_position,
        )
        if np.allclose(position, shape.position, atol=1e-4):
            return False
        shape.position = position
        return True

    def begin_transform(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        minimum_distance: float = 24.0,
        first_depth_signal: float = 0.0,
        second_depth_signal: float = 0.0,
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
            start_depth_balance=_depth_balance(
                first_depth_signal,
                second_depth_signal,
            ),
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
        if not 0 < minimum_scale <= maximum_scale:
            raise ValueError("Expected 0 < minimum_scale <= maximum_scale")
        if self._move is None or self._move.transform is None:
            return False

        shape = self.objects[self._move.object_index]
        transform = self._move.transform
        distance = hypot(second[0] - first[0], second[1] - first[1])
        scale_ratio = distance / transform.start_distance
        if abs(scale_ratio - 1.0) < 0.035:
            scale_ratio = 1.0
        scale = float(
            np.clip(
                transform.base_scale * scale_ratio,
                minimum_scale,
                maximum_scale,
            )
        )
        angle = atan2(second[1] - first[1], second[0] - first[0])
        spin_delta = _normalize_angle(degrees(angle - transform.start_angle))
        if abs(spin_delta) < 2.5:
            spin_delta = 0.0
        rotation_z = _normalize_angle(transform.base_rotation_z + spin_delta)
        midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        midpoint_dx = _dead_zone(midpoint[0] - transform.start_midpoint[0], 5.0)
        midpoint_dy = _dead_zone(midpoint[1] - transform.start_midpoint[1], 5.0)
        depth_tilt = 0.0
        current_balance = _depth_balance(first_depth_signal, second_depth_signal)
        if transform.start_depth_balance is not None and current_balance is not None:
            depth_tilt = (current_balance - transform.start_depth_balance) * 120.0

        rotation_x = float(
            np.clip(transform.base_rotation_x + midpoint_dy * 0.9, -78.0, 78.0)
        )
        rotation_y = float(
            np.clip(
                transform.base_rotation_y + midpoint_dx * 0.65 + depth_tilt,
                -78.0,
                78.0,
            )
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

    def end_transform(
        self,
        remaining_anchor: tuple[int, int] | None = None,
        remaining_depth_signal: float = 0.0,
    ) -> bool:
        if self._move is None or self._move.transform is None:
            return False
        self._move.transform = None
        if remaining_anchor is not None:
            shape = self.objects[self._move.object_index]
            self._move.anchor = remaining_anchor
            self._move.base_position = shape.position
            self._move.start_depth_signal = (
                remaining_depth_signal if remaining_depth_signal > 1e-6 else None
            )
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

    def adjust_z(self, change: float) -> bool:
        shape = self._selected()
        if shape is None:
            return False
        x, y, current_z = shape.position
        z_position = self._clamp_z(current_z + change)
        if abs(z_position - current_z) < 1e-6:
            return False
        shape.position = x, y, z_position
        return True

    def render(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Frame and scene dimensions do not match")
        ordered = sorted(self.objects, key=lambda shape: shape.position[2])
        for shape in ordered:
            self._render_shape(frame, shape)

    def _selected(self) -> SpatialShape | None:
        if self.selected_index is None or self.selected_index >= len(self.objects):
            return None
        return self.objects[self.selected_index]

    def _hit_test(self, shape: SpatialShape, point: tuple[int, int]) -> bool:
        projected, _ = self._project(shape)
        return cv2.pointPolygonTest(projected, point, False) >= 0

    def _render_shape(self, frame: np.ndarray, shape: SpatialShape) -> None:
        projected, normal_z = self._project(shape)
        polygon = np.rint(projected).astype(np.int32)
        face_shade = 0.62 + 0.38 * abs(normal_z)
        cv2.fillPoly(frame, [polygon], _shade(shape.color, face_shade), cv2.LINE_AA)
        cv2.polylines(
            frame,
            [polygon],
            True,
            _shade(shape.color, 0.35),
            2,
            cv2.LINE_AA,
        )

    def _project(self, shape: SpatialShape) -> tuple[np.ndarray, float]:
        count = len(shape.contour)
        vertices = np.column_stack(
            (shape.contour, np.zeros(count, dtype=np.float32))
        )
        rotation = _rotation_matrix(
            shape.rotation_x,
            shape.rotation_y,
            shape.rotation_z,
        )
        projected = self._project_local_points(shape, vertices, rotation)
        normal_z = float((rotation @ np.asarray((0, 0, 1), dtype=np.float32))[2])
        return projected, normal_z

    def _project_local_points(
        self,
        shape: SpatialShape,
        points: np.ndarray,
        rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        if rotation is None:
            rotation = _rotation_matrix(
                shape.rotation_x,
                shape.rotation_y,
                shape.rotation_z,
            )
        rotated = (np.asarray(points, dtype=np.float32) * shape.scale) @ rotation.T
        camera_z = shape.position[2] + rotated[:, 2]
        camera_distance = np.maximum(
            self.focal_length - camera_z,
            self.focal_length * 0.08,
        )
        perspective = np.clip(
            self.focal_length / camera_distance,
            0.15,
            4.0,
        )
        projected = np.column_stack(
            (
                shape.position[0] + rotated[:, 0] * perspective,
                shape.position[1] + rotated[:, 1] * perspective,
            )
        ).astype(np.float32)
        return projected

    def _depth_from_signal(self, depth_signal: float) -> float:
        if (
            self._move is None
            or self._move.start_depth_signal is None
            or depth_signal <= 1e-6
        ):
            return self._move.base_position[2] if self._move is not None else 0.0
        ratio_log = log(depth_signal / self._move.start_depth_signal)
        if abs(ratio_log) <= self.depth_dead_zone:
            ratio_log = 0.0
        else:
            ratio_log -= np.sign(ratio_log) * self.depth_dead_zone
        delta = ratio_log * self.focal_length * self.depth_sensitivity
        return self._clamp_z(self._move.base_position[2] + delta)

    def _clamp_z(self, value: float) -> float:
        minimum_z, maximum_z = self.z_limits
        return float(np.clip(value, minimum_z, maximum_z))


def _rotation_matrix(x_degrees: float, y_degrees: float, z_degrees: float) -> np.ndarray:
    x, y, z = map(radians, (x_degrees, y_degrees, z_degrees))
    rx = np.asarray(((1, 0, 0), (0, cos(x), -sin(x)), (0, sin(x), cos(x))))
    ry = np.asarray(((cos(y), 0, sin(y)), (0, 1, 0), (-sin(y), 0, cos(y))))
    rz = np.asarray(((cos(z), -sin(z), 0), (sin(z), cos(z), 0), (0, 0, 1)))
    return (rz @ ry @ rx).astype(np.float32)


def _normalize_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _dead_zone(value: float, threshold: float) -> float:
    if abs(value) <= threshold:
        return 0.0
    return value - np.sign(value) * threshold


def _depth_balance(first: float, second: float) -> float | None:
    if first <= 1e-6 or second <= 1e-6:
        return None
    return log(first / second)


def _shade(color: Color, factor: float) -> Color:
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)
