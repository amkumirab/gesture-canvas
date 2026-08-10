"""Stateful coordination for one- and two-hand canvas manipulation."""

from __future__ import annotations

from dataclasses import dataclass

from .canvas import DrawingCanvas


@dataclass(frozen=True, slots=True)
class PinchHand:
    key: str
    cursor: tuple[int, int]
    pinching: bool
    started: bool = False


class GrabCoordinator:
    """Translate with one pinch and scale/rotate with two pinches."""

    def __init__(self, selection_radius: int = 32) -> None:
        if selection_radius < 0:
            raise ValueError("selection_radius cannot be negative")
        self.selection_radius = selection_radius
        self.hand_ids: list[str] = []

    def update(
        self,
        canvas: DrawingCanvas,
        hands: list[PinchHand],
        canvas_top: int = 0,
    ) -> bool:
        by_key = {hand.key: hand for hand in hands}

        if not canvas.is_moving:
            self.hand_ids.clear()
            for hand in hands:
                if (
                    hand.pinching
                    and hand.started
                    and hand.cursor[1] > canvas_top
                    and canvas.begin_move(hand.cursor, self.selection_radius)
                ):
                    self.hand_ids = [hand.key]
                    return True
            return False

        active = [
            by_key[key]
            for key in self.hand_ids
            if key in by_key and by_key[key].pinching
        ]
        if not active:
            canvas.end_move()
            self.hand_ids.clear()
            return False

        if len(active) >= 2:
            first, second = active[:2]
            if not canvas.is_transforming:
                canvas.begin_transform(first.cursor, second.cursor)
            if canvas.is_transforming:
                canvas.update_transform(first.cursor, second.cursor)
            self.hand_ids = [first.key, second.key]
            return True

        primary = active[0]
        if canvas.is_transforming:
            canvas.end_transform(primary.cursor)
        self.hand_ids = [primary.key]

        candidates = [
            hand
            for hand in hands
            if hand.key != primary.key and hand.pinching
        ]
        for second in candidates:
            if canvas.begin_transform(primary.cursor, second.cursor):
                self.hand_ids.append(second.key)
                canvas.update_transform(primary.cursor, second.cursor)
                return True

        canvas.update_move(primary.cursor)
        return True

    def reset(self, canvas: DrawingCanvas | None = None) -> None:
        if canvas is not None:
            canvas.end_move()
        self.hand_ids.clear()
