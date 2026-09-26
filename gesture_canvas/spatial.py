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
    visible: bool = True
    locked: bool = False

    def copy(self) -> SpatialShape:
        return SpatialShape(
            contour=self.contour.copy(),
            color=self.color,
            position=self.position,
            scale=self.scale,
            rotation_x=self.rotation_x,
            rotation_y=self.rotation_y,
            rotation_z=self.rotation_z,
            visible=self.visible,
            locked=self.locked,
        )


@dataclass(frozen=True, slots=True)
class SpatialLayerInfo:
    index: int
    color: Color
    position: tuple[float, float, float]
    scale: float
    rotation_x: float
    rotation_y: float
    rotation_z: float
    visible: bool
    locked: bool


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
    base_scale: float
    base_rotation_x: float
    base_rotation_y: float
    base_rotation_z: float
    start_depth_balance: float | None
    previous_angle: float
    accumulated_spin: float = 0.0
    order_reversed: bool = False
    control: str = "waiting"
    candidate: str = "waiting"
    candidate_frames: int = 0
    decision_frames: int = 0


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

    def __init__(
        self,
        width: int,
        height: int,
        depth_sensitivity: float = 1.35,
    ) -> None:
        if not 0.5 <= depth_sensitivity <= 2.5:
            raise ValueError("depth_sensitivity must be between 0.5 and 2.5")
        self.width = width
        self.height = height
        self.depth_sensitivity = depth_sensitivity
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
        if shape is None or not shape.visible:
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
    def transform_control(self) -> str | None:
        if self._move is None or self._move.transform is None:
            return None
        return self._move.transform.control

    @property
    def selected_z(self) -> float | None:
        shape = self._selected()
        return None if shape is None else shape.position[2]

    def selected_guide(self, axis_length: float = 54.0) -> SpatialGuide | None:
        """Return projected local axes and depth range for the selected shape."""

        if axis_length <= 0:
            raise ValueError("axis_length must be positive")
        shape = self._selected()
        if shape is None or not shape.visible:
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

    def layer_info(self) -> tuple[SpatialLayerInfo, ...]:
        """Return detached metadata used by the layer-management panel."""

        return tuple(
            SpatialLayerInfo(
                index=index,
                color=shape.color,
                position=shape.position,
                scale=shape.scale,
                rotation_x=shape.rotation_x,
                rotation_y=shape.rotation_y,
                rotation_z=shape.rotation_z,
                visible=shape.visible,
                locked=shape.locked,
            )
            for index, shape in enumerate(self.objects)
        )

    def select_layer(self, index: int) -> bool:
        if not 0 <= index < len(self.objects) or index == self.selected_index:
            return False
        self.end_move()
        self.selected_index = index
        return True

    def toggle_selected_visibility(self) -> bool:
        shape = self._selected()
        if shape is None:
            return False
        self.end_move()
        shape.visible = not shape.visible
        return True

    def toggle_selected_lock(self) -> bool:
        shape = self._selected()
        if shape is None:
            return False
        self.end_move()
        shape.locked = not shape.locked
        return True

    def duplicate_selected(self, offset: float = 18.0) -> bool:
        shape = self._selected()
        if shape is None:
            return False
        self.end_move()
        duplicate = shape.copy()
        x, y, z = duplicate.position
        duplicate.position = (x + offset, y + offset, z)
        duplicate.visible = True
        duplicate.locked = False
        self.objects.append(duplicate)
        self.selected_index = len(self.objects) - 1
        return True

    def delete_selected(self) -> bool:
        if self.selected_index is None or not self.objects:
            return False
        self.end_move()
        removed = self.selected_index
        del self.objects[removed]
        self.selected_index = (
            min(removed, len(self.objects) - 1) if self.objects else None
        )
        return True

    def reorder_selected(self, change: int) -> bool:
        """Move the selected layer in stacking order by one or more positions."""

        if self.selected_index is None or change == 0:
            return False
        target = int(np.clip(self.selected_index + change, 0, len(self.objects) - 1))
        if target == self.selected_index:
            return False
        self.end_move()
        shape = self.objects.pop(self.selected_index)
        self.objects.insert(target, shape)
        self.selected_index = target
        return True

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
            shape = self.objects[index]
            if shape.visible and not shape.locked and self._hit_test(shape, point):
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
        angle = atan2(second[1] - first[1], second[0] - first[0])
        self._move.transform = SpatialTransform(
            start_midpoint=midpoint,
            start_distance=distance,
            base_scale=shape.scale,
            base_rotation_x=shape.rotation_x,
            base_rotation_y=shape.rotation_y,
            base_rotation_z=shape.rotation_z,
            start_depth_balance=_depth_balance(
                first_depth_signal,
                second_depth_signal,
            ),
            previous_angle=angle,
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
        angle = atan2(second[1] - first[1], second[0] - first[0])
        angle_step, order_swapped = _stable_angle_step(
            transform.previous_angle,
            angle,
        )
        transform.previous_angle = angle
        if order_swapped:
            transform.order_reversed = not transform.order_reversed
        transform.accumulated_spin += angle_step
        spin_delta = transform.accumulated_spin
        midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        midpoint_dx = midpoint[0] - transform.start_midpoint[0]
        midpoint_dy = midpoint[1] - transform.start_midpoint[1]
        depth_delta = 0.0
        current_balance = _depth_balance(first_depth_signal, second_depth_signal)
        if current_balance is not None and transform.order_reversed:
            current_balance = -current_balance
        if transform.start_depth_balance is not None and current_balance is not None:
            depth_delta = current_balance - transform.start_depth_balance

        tilt_x_delta = midpoint_dy * 0.85
        tilt_y_delta = midpoint_dx * 0.60 + depth_delta * 115.0
        if transform.control == "waiting":
            transform.decision_frames += 1
            scores = _transform_control_scores(
                scale_ratio,
                spin_delta,
                tilt_x_delta,
                tilt_y_delta,
            )
            candidate = _choose_transform_control(scores)
            if candidate == "waiting" and transform.decision_frames >= 8:
                best_control, best_score = max(
                    scores.items(),
                    key=lambda item: item[1],
                )
                if best_score >= 1.15:
                    candidate = best_control
            if candidate == "waiting":
                transform.candidate = "waiting"
                transform.candidate_frames = 0
                return False
            if candidate == transform.candidate:
                transform.candidate_frames += 1
            else:
                transform.candidate = candidate
                transform.candidate_frames = 1
            candidate_score = scores[candidate]
            if (
                candidate_score < 2.2
                and transform.candidate_frames < 2
                and transform.decision_frames < 8
            ):
                return False
            transform.control = candidate

        scale = shape.scale
        rotation_x = shape.rotation_x
        rotation_y = shape.rotation_y
        rotation_z = shape.rotation_z
        if transform.control == "scale":
            target = float(
                np.clip(
                    transform.base_scale * scale_ratio,
                    minimum_scale,
                    maximum_scale,
                )
            )
            scale = _smooth_value(shape.scale, target, 0.52)
        elif transform.control == "spin":
            target = _normalize_angle(transform.base_rotation_z + spin_delta)
            rotation_z = _smooth_angle(shape.rotation_z, target, 0.48)
        elif transform.control == "tilt_x":
            target = float(
                np.clip(transform.base_rotation_x + tilt_x_delta, -72.0, 72.0)
            )
            rotation_x = _smooth_value(shape.rotation_x, target, 0.44)
        elif transform.control == "tilt_y":
            target = float(
                np.clip(transform.base_rotation_y + tilt_y_delta, -72.0, 72.0)
            )
            rotation_y = _smooth_value(shape.rotation_y, target, 0.44)
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
        if shape is None or shape.locked:
            return False
        x, y, current_z = shape.position
        z_position = self._clamp_z(current_z + change)
        if abs(z_position - current_z) < 1e-6:
            return False
        shape.position = x, y, z_position
        return True

    def reset_selected_rotation(self) -> bool:
        """Restore the selected layer to a front-facing orientation."""

        shape = self._selected()
        if shape is None or shape.locked:
            return False
        if np.allclose(
            (shape.rotation_x, shape.rotation_y, shape.rotation_z),
            (0.0, 0.0, 0.0),
            atol=1e-4,
        ):
            return False
        shape.rotation_x = 0.0
        shape.rotation_y = 0.0
        shape.rotation_z = 0.0
        return True

    def render(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("Frame and scene dimensions do not match")
        ordered = sorted(
            enumerate(self.objects),
            key=lambda item: (item[1].position[2], item[0]),
        )
        for _, shape in ordered:
            if shape.visible:
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


def _transform_control_scores(
    scale_ratio: float,
    spin_delta: float,
    tilt_x_delta: float,
    tilt_y_delta: float,
) -> dict[str, float]:
    """Return normalized intent scores with a slight rotation preference."""

    safe_scale = max(scale_ratio, 1e-6)
    return {
        "scale": abs(log(safe_scale)) / 0.10,
        "spin": abs(spin_delta) / 6.0,
        "tilt_x": abs(tilt_x_delta) / 14.0,
        "tilt_y": abs(tilt_y_delta) / 14.0,
    }


def _choose_transform_control(scores: dict[str, float]) -> str:
    """Choose only a clear gesture so early tracking noise cannot lock a mode."""

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (control, score), (_, runner_up) = ordered[:2]
    if score < 1.0 or score < runner_up * 1.18:
        return "waiting"
    return control


def _stable_angle_step(previous: float, current: float) -> tuple[float, bool]:
    """Return a continuous angle step and absorb an endpoint-order swap."""

    step = _normalize_angle(degrees(current - previous))
    swapped = abs(step) > 100.0
    if swapped:
        step -= 180.0 if step > 0 else -180.0
    return step, swapped


def _smooth_value(current: float, target: float, alpha: float) -> float:
    value = current + (target - current) * alpha
    return target if abs(target - value) < 1e-3 else value


def _smooth_angle(current: float, target: float, alpha: float) -> float:
    delta = _normalize_angle(target - current)
    if abs(delta) < 1e-3:
        return target
    return _normalize_angle(current + delta * alpha)


def _depth_balance(first: float, second: float) -> float | None:
    if first <= 1e-6 or second <= 1e-6:
        return None
    return log(first / second)


def _shade(color: Color, factor: float) -> Color:
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)
