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
    depth_signal: float = 0.0


class GrabCoordinator:
    """Translate with one pinch and scale/rotate with two pinches."""

    def __init__(
        self,
        selection_radius: int = 32,
        missing_grace_frames: int = 8,
    ) -> None:
        if selection_radius < 0:
            raise ValueError("selection_radius cannot be negative")
        if missing_grace_frames < 0:
            raise ValueError("missing_grace_frames cannot be negative")
        self.selection_radius = selection_radius
        self.missing_grace_frames = missing_grace_frames
        self.hand_ids: list[str] = []
        self._last_hands: dict[str, PinchHand] = {}
        self._missing_frames: dict[str, int] = {}

    @property
    def recovering_tracking(self) -> bool:
        """Whether an active hand is temporarily being recovered after dropout."""

        return any(
            key in self._missing_frames and self._missing_frames[key] > 0
            for key in self.hand_ids
        )

    def update(
        self,
        canvas: DrawingCanvas,
        hands: list[PinchHand],
        canvas_top: int = 0,
    ) -> bool:
        by_key = {hand.key: hand for hand in hands}
        for hand in hands:
            if hand.pinching:
                self._last_hands[hand.key] = hand
                self._missing_frames.pop(hand.key, None)

        if not canvas.is_moving:
            self.hand_ids.clear()
            for hand in hands:
                if (
                    hand.pinching
                    and hand.started
                    and hand.cursor[1] > canvas_top
                    and canvas.begin_move(
                        hand.cursor,
                        self.selection_radius,
                        hand.depth_signal,
                    )
                ):
                    self.hand_ids = [hand.key]
                    self._last_hands[hand.key] = hand
                    return True
            return False

        active: list[PinchHand] = []
        for key in self.hand_ids:
            current = by_key.get(key)
            if current is not None:
                self._missing_frames.pop(key, None)
                if current.pinching:
                    self._last_hands[key] = current
                    active.append(current)
                else:
                    self._last_hands.pop(key, None)
                continue

            missing = self._missing_frames.get(key, 0) + 1
            self._missing_frames[key] = missing
            cached = self._last_hands.get(key)
            if cached is not None and missing <= self.missing_grace_frames:
                active.append(cached)
            else:
                self._last_hands.pop(key, None)
                self._missing_frames.pop(key, None)

        if not active:
            canvas.end_move()
            self.hand_ids.clear()
            return False

        if len(active) >= 2:
            first, second = active[:2]
            if not canvas.is_transforming:
                canvas.begin_transform(
                    first.cursor,
                    second.cursor,
                    first_depth_signal=first.depth_signal,
                    second_depth_signal=second.depth_signal,
                )
            if canvas.is_transforming:
                canvas.update_transform(
                    first.cursor,
                    second.cursor,
                    first_depth_signal=first.depth_signal,
                    second_depth_signal=second.depth_signal,
                )
            self.hand_ids = [first.key, second.key]
            return True

        primary = active[0]
        if canvas.is_transforming:
            canvas.end_transform(primary.cursor, primary.depth_signal)
        self.hand_ids = [primary.key]

        candidates = [
            hand
            for hand in hands
            if hand.key != primary.key and hand.pinching
        ]
        for second in candidates:
            if canvas.begin_transform(
                primary.cursor,
                second.cursor,
                first_depth_signal=primary.depth_signal,
                second_depth_signal=second.depth_signal,
            ):
                self.hand_ids.append(second.key)
                self._last_hands[second.key] = second
                canvas.update_transform(
                    primary.cursor,
                    second.cursor,
                    first_depth_signal=primary.depth_signal,
                    second_depth_signal=second.depth_signal,
                )
                return True

        canvas.update_move(primary.cursor, primary.depth_signal)
        return True

    def reset(self, canvas: DrawingCanvas | None = None) -> None:
        if canvas is not None:
            canvas.end_move()
        self.hand_ids.clear()
        self._last_hands.clear()
        self._missing_frames.clear()
