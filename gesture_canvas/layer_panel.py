"""On-canvas management panel for spatial drawing layers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .spatial import SpatialLayerInfo


@dataclass(frozen=True, slots=True)
class LayerPanelTarget:
    key: str
    label: str


class LayerPanel:
    """Render and hit-test a compact layer panel controlled by dwell pointing."""

    margin = 8
    row_height = 38
    header_height = 42
    footer_height = 128
    gap = 5

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        content_top: int = 74,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Panel dimensions must be positive")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.top = min(content_top + 8, max(8, frame_height - 90))
        self.right = frame_width - self.margin
        self.width = min(310, max(230, frame_width // 3))
        self.left = max(self.margin, self.right - self.width)
        self.bottom = frame_height - self.margin

    def contains(self, point: tuple[int, int]) -> bool:
        x, y = point
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def hit_test(
        self,
        point: tuple[int, int],
        layers: tuple[SpatialLayerInfo, ...],
        selected_index: int | None,
    ) -> LayerPanelTarget | None:
        if not self.contains(point):
            return None
        for index, rect in self._layer_rects(len(layers), selected_index):
            if _inside(point, rect):
                return LayerPanelTarget(f"select:{index}", f"Layer {index + 1}")
        if selected_index is None or not 0 <= selected_index < len(layers):
            return None
        selected = layers[selected_index]
        for key, label, rect in self._action_rects(selected):
            if _inside(point, rect):
                return LayerPanelTarget(key, label)
        return None

    def draw(
        self,
        frame: np.ndarray,
        layers: tuple[SpatialLayerInfo, ...],
        selected_index: int | None,
        hover_key: str | None = None,
        hover_progress: float = 0.0,
    ) -> None:
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            raise ValueError("Frame and layer panel dimensions do not match")

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (self.left, self.top),
            (self.right, self.bottom),
            (15, 19, 27),
            -1,
        )
        cv2.addWeighted(overlay, 0.91, frame, 0.09, 0, frame)
        cv2.rectangle(
            frame,
            (self.left, self.top),
            (self.right, self.bottom),
            (80, 220, 255),
            2,
        )
        cv2.putText(
            frame,
            f"LAYERS  {len(layers)}   [P] CLOSE",
            (self.left + 12, self.top + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )

        if not layers:
            cv2.putText(
                frame,
                "CREATE A 3D LAYER FIRST",
                (self.left + 12, self.top + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (190, 198, 210),
                1,
                cv2.LINE_AA,
            )
            return

        visible_rects = self._layer_rects(len(layers), selected_index)
        for index, rect in visible_rects:
            layer = layers[index]
            target_key = f"select:{index}"
            selected = index == selected_index
            hovered = target_key == hover_key
            fill = (43, 50, 62) if layer.visible else (29, 33, 42)
            border = (
                (80, 220, 255)
                if selected or hovered
                else (78, 86, 100)
            )
            _draw_button(frame, rect, fill, border, target_key, hover_key, hover_progress)
            left, top, _, bottom = rect
            cv2.rectangle(
                frame,
                (left + 8, top + 9),
                (left + 26, bottom - 9),
                tuple(int(channel) for channel in layer.color),
                -1,
            )
            state = "HIDDEN" if not layer.visible else "LOCKED" if layer.locked else "READY"
            cv2.putText(
                frame,
                f"LAYER {index + 1:02d}",
                (left + 35, top + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (245, 247, 250),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                state,
                (left + 35, top + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (150, 160, 174),
                1,
                cv2.LINE_AA,
            )

        if selected_index is None or not 0 <= selected_index < len(layers):
            return
        selected = layers[selected_index]
        detail_y = self.bottom - self.footer_height + 18
        x, y, z = selected.position
        cv2.putText(
            frame,
            f"X {x:.0f}  Y {y:.0f}  Z {z:+.0f}",
            (self.left + 10, detail_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (225, 230, 237),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            (
                f"S {selected.scale:.2f}  RX {selected.rotation_x:+.0f}  "
                f"RY {selected.rotation_y:+.0f}  RZ {selected.rotation_z:+.0f}"
            ),
            (self.left + 10, detail_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (180, 190, 202),
            1,
            cv2.LINE_AA,
        )
        for key, label, rect in self._action_rects(selected):
            hovered = key == hover_key
            border = (80, 220, 255) if hovered else (78, 86, 100)
            fill = (48, 55, 68)
            if key == "delete":
                border = (80, 110, 240) if hovered else (70, 80, 145)
            _draw_button(frame, rect, fill, border, key, hover_key, hover_progress)
            _centered_text(frame, label, rect)

    def _layer_rects(
        self,
        count: int,
        selected_index: int | None,
    ) -> list[tuple[int, tuple[int, int, int, int]]]:
        rows_top = self.top + self.header_height
        rows_bottom = self.bottom - self.footer_height
        capacity = max(1, (rows_bottom - rows_top) // self.row_height)
        order = list(reversed(range(count)))
        if len(order) > capacity and selected_index in order:
            selected_position = order.index(selected_index)
            start = min(
                max(0, selected_position - capacity // 2),
                len(order) - capacity,
            )
            order = order[start : start + capacity]
        else:
            order = order[:capacity]
        result = []
        for row, index in enumerate(order):
            top = rows_top + row * self.row_height
            result.append(
                (index, (self.left + 7, top, self.right - 7, top + self.row_height - 4))
            )
        return result

    def _action_rects(
        self,
        selected: SpatialLayerInfo,
    ) -> list[tuple[str, str, tuple[int, int, int, int]]]:
        actions = (
            ("visibility", "SHOW" if not selected.visible else "HIDE"),
            ("lock", "UNLOCK" if selected.locked else "LOCK"),
            ("duplicate", "COPY"),
            ("lower", "LOWER"),
            ("raise", "RAISE"),
            ("delete", "DELETE"),
        )
        left = self.left + 8
        right = self.right - 8
        top = self.bottom - 70
        button_width = (right - left - 2 * self.gap) // 3
        button_height = 29
        rects = []
        for index, (key, label) in enumerate(actions):
            row, column = divmod(index, 3)
            x1 = left + column * (button_width + self.gap)
            y1 = top + row * (button_height + self.gap)
            rects.append((key, label, (x1, y1, x1 + button_width, y1 + button_height)))
        return rects


def _inside(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def _draw_button(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    border: tuple[int, int, int],
    key: str,
    hover_key: str | None,
    hover_progress: float,
) -> None:
    left, top, right, bottom = rect
    cv2.rectangle(frame, (left, top), (right, bottom), fill, -1)
    cv2.rectangle(frame, (left, top), (right, bottom), border, 2)
    if key == hover_key and hover_progress > 0:
        progress_right = left + round((right - left) * min(1.0, hover_progress))
        cv2.rectangle(
            frame,
            (left + 2, bottom - 5),
            (progress_right, bottom - 2),
            border,
            -1,
        )


def _centered_text(
    frame: np.ndarray,
    label: str,
    rect: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = rect
    font_scale = 0.36 if right - left < 90 else 0.42
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    x = left + max(3, (right - left - size[0]) // 2)
    y = top + (bottom - top + size[1]) // 2
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (242, 245, 248),
        1,
        cv2.LINE_AA,
    )
