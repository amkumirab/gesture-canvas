"""Real-time Gesture Canvas application."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from .camera import DetectedHand, create_hand_tracker, detect_hands, open_camera
from .canvas import DrawingCanvas
from .controls import DEFAULT_BRUSH_SIZE, adjust_brush_size
from .gestures import (
    DrawGestureStabilizer,
    Gesture,
    PinchDetector,
    RuleBasedGestureClassifier,
)
from .help_overlay import draw_help_overlay
from .interaction import GrabCoordinator, PinchHand
from .landmarks import to_pixel
from .model import NeuralGesturePredictor
from .smoothing import AdaptiveSmoother
from .toolbar import DwellSelector, Toolbar, ToolButton


COLORS = {
    "blue": (255, 90, 30),
    "green": (60, 210, 70),
    "red": (50, 60, 240),
    "black": (20, 20, 20),
}


@dataclass(frozen=True, slots=True)
class HandFrame:
    key: str
    cursor: tuple[int, int]
    gesture: Gesture
    confidence: float
    pinching: bool
    pinch_started: bool
    pinch_ended: bool
    handedness: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw in the air using hand gestures.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=Path, help="Optional trained .pt gesture model")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Saved image folder")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_canvas(canvas: DrawingCanvas, output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return canvas.export(output_dir / f"gesture_canvas_{stamp}.png")


def apply_button(
    button: ToolButton,
    canvas: DrawingCanvas,
    output_dir: Path,
    active_tool: str,
) -> tuple[str, str]:
    if button.key in COLORS:
        return button.key, f"Pen color: {button.label.lower()}"
    if button.key == "eraser":
        return "eraser", "Eraser selected"
    if button.key == "undo":
        return active_tool, "Undo" if canvas.undo() else "Nothing to undo"
    if button.key == "redo":
        return active_tool, "Redo" if canvas.redo() else "Nothing to redo"
    if button.key == "clear":
        canvas.clear()
        return active_tool, "Canvas cleared"
    if button.key == "save":
        path = save_canvas(canvas, output_dir)
        return active_tool, f"Saved: {path.name}"
    return active_tool, ""


def hand_key(hand: DetectedHand, index: int, used: set[str]) -> str:
    """Build a stable key from MediaPipe handedness with a safe fallback."""

    base = hand.handedness if hand.handedness in {"left", "right"} else "hand"
    key = base
    if key in used:
        key = f"{base}-{index}"
    used.add(key)
    return key


def main() -> None:
    args = parse_args()
    classifier = (
        NeuralGesturePredictor(args.model)
        if args.model
        else RuleBasedGestureClassifier()
    )
    camera = open_camera(args.camera)
    try:
        tracker = create_hand_tracker()
    except Exception:
        camera.release()
        raise

    pinch_detectors: dict[str, PinchDetector] = {}
    smoothers: dict[str, AdaptiveSmoother] = {}
    draw_stabilizers: dict[str, DrawGestureStabilizer] = {}
    grab_coordinator = GrabCoordinator(selection_radius=32)

    canvas: DrawingCanvas | None = None
    toolbar: Toolbar | None = None
    selector = DwellSelector(dwell_seconds=0.7)
    active_tool = "blue"
    brush_size = DEFAULT_BRUSH_SIZE
    show_help = False
    drawing_hand_id: str | None = None
    hover_key: str | None = None
    hover_progress = 0.0
    status = "Pinch with one hand to move; pinch with two hands to scale and rotate"
    status_until = time.monotonic() + 5
    draw_blocked_until = 0.0
    previous_time = time.monotonic()
    fps = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames")
            if args.mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            if canvas is None:
                canvas = DrawingCanvas(width, height)
                toolbar = Toolbar(width)
            assert toolbar is not None

            interaction_time = time.monotonic()
            detected = detect_hands(tracker, frame)
            hands: list[HandFrame] = []
            used_keys: set[str] = set()
            seen_keys: set[str] = set()

            for index, detected_hand in enumerate(detected):
                key = hand_key(detected_hand, index, used_keys)
                seen_keys.add(key)
                pinch_detector = pinch_detectors.setdefault(key, PinchDetector())
                smoother = smoothers.setdefault(key, AdaptiveSmoother())
                stabilizer = draw_stabilizers.setdefault(key, DrawGestureStabilizer())
                raw_gesture, confidence = classifier.classify(detected_hand.landmarks)
                pinching, pinch_started, pinch_ended, _ = pinch_detector.update(
                    detected_hand.landmarks
                )
                gesture = stabilizer.update(raw_gesture, interaction_time)
                if pinching:
                    gesture = Gesture.PINCH
                if pinch_ended:
                    draw_blocked_until = interaction_time + 0.22
                cursor = smoother.update(
                    to_pixel(detected_hand.landmarks[8], width, height)
                )
                hands.append(
                    HandFrame(
                        key=key,
                        cursor=cursor,
                        gesture=gesture,
                        confidence=confidence,
                        pinching=pinching,
                        pinch_started=pinch_started,
                        pinch_ended=pinch_ended,
                        handedness=detected_hand.handedness,
                    )
                )

            for key, detector in pinch_detectors.items():
                if key not in seen_keys:
                    if detector.active:
                        draw_blocked_until = interaction_time + 0.22
                    detector.reset()
                    smoothers[key].reset()

            pinch_hands = [
                PinchHand(
                    key=hand.key,
                    cursor=hand.cursor,
                    pinching=hand.pinching,
                    started=hand.pinch_started,
                )
                for hand in hands
            ]
            was_moving = canvas.is_moving
            grab_coordinator.update(canvas, pinch_hands, canvas_top=toolbar.height)
            manipulation_frame = (
                was_moving
                or canvas.is_moving
                or any(hand.pinching for hand in hands)
            )

            hover_key = None
            hover_progress = 0.0
            display_gesture = Gesture.PINCH if manipulation_frame else Gesture.IDLE
            display_confidence = 0.0

            if manipulation_frame:
                canvas.end_stroke()
                selector.reset()
            else:
                available = [hand for hand in hands if not hand.pinching]
                active_hand = next(
                    (hand for hand in available if hand.key == drawing_hand_id),
                    None,
                )
                if active_hand is None and available:
                    active_hand = max(available, key=lambda hand: hand.confidence)
                    if drawing_hand_id != active_hand.key:
                        canvas.end_stroke()
                        selector.reset()
                    drawing_hand_id = active_hand.key

                if active_hand is not None:
                    display_gesture = active_hand.gesture
                    display_confidence = active_hand.confidence
                    in_toolbar = active_hand.cursor[1] <= toolbar.height
                    if in_toolbar:
                        canvas.end_stroke()
                        can_select = active_hand.gesture is Gesture.DRAW
                        hovered = (
                            toolbar.hit_test(active_hand.cursor) if can_select else None
                        )
                        activated, hover_progress = selector.update(
                            hovered, interaction_time
                        )
                        hover_key = hovered.key if hovered else None
                        if activated:
                            active_tool, status = apply_button(
                                activated, canvas, args.output, active_tool
                            )
                            status_until = interaction_time + 2
                    else:
                        selector.reset()
                        drawing = (
                            active_hand.gesture is Gesture.DRAW
                            and interaction_time >= draw_blocked_until
                        )
                        two_finger_erase = active_hand.gesture is Gesture.ERASE
                        if drawing:
                            if active_tool == "eraser":
                                canvas.erase_point(
                                    active_hand.cursor,
                                    46,
                                    max_segment_length=110,
                                )
                            else:
                                canvas.add_point(
                                    active_hand.cursor,
                                    COLORS[active_tool],
                                    brush_size,
                                    max_segment_length=90,
                                )
                        elif two_finger_erase:
                            canvas.erase_point(
                                active_hand.cursor,
                                46,
                                max_segment_length=110,
                            )
                        else:
                            canvas.end_stroke()
                else:
                    selector.reset()
                    if drawing_hand_id is None or not draw_stabilizers[
                        drawing_hand_id
                    ].hold_during_missing(interaction_time):
                        canvas.end_stroke()

            display = canvas.composite(frame)
            if canvas.move_bounds:
                x, y, box_width, box_height = canvas.move_bounds
                cv2.rectangle(
                    display,
                    (x - 5, y - 5),
                    (x + box_width + 5, y + box_height + 5),
                    (80, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                if canvas.is_transforming and canvas.transform_info:
                    scale, rotation = canvas.transform_info
                    cv2.putText(
                        display,
                        f"{scale:.2f}x  {rotation:+.0f} deg",
                        (max(8, x), max(toolbar.height + 24, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (80, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )

            hands_by_key = {hand.key: hand for hand in hands}
            if canvas.is_transforming and len(grab_coordinator.hand_ids) >= 2:
                first = hands_by_key.get(grab_coordinator.hand_ids[0])
                second = hands_by_key.get(grab_coordinator.hand_ids[1])
                if first and second:
                    cv2.line(
                        display,
                        first.cursor,
                        second.cursor,
                        (255, 170, 60),
                        2,
                        cv2.LINE_AA,
                    )

            toolbar.draw(display, active_tool, hover_key, hover_progress)
            for hand in hands:
                is_drawing_hand = hand.key == drawing_hand_id
                radius = 14 if hand.pinching else 9
                if is_drawing_hand and not hand.pinching and (
                    hand.gesture is Gesture.ERASE or active_tool == "eraser"
                ):
                    radius = 23
                elif is_drawing_hand and not hand.pinching:
                    radius = max(4, round(brush_size / 2))
                color = (255, 255, 255) if hand.pinching else (80, 220, 255)
                if not is_drawing_hand and not hand.pinching:
                    color = (255, 170, 60)
                cv2.circle(display, hand.cursor, radius, color, 2, cv2.LINE_AA)
                label = hand.handedness[:1].upper() or "H"
                cv2.putText(
                    display,
                    label,
                    (hand.cursor[0] + 12, hand.cursor[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            now = time.monotonic()
            elapsed = max(now - previous_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
            previous_time = now
            label = (
                f"{display_gesture.value}  {display_confidence:.0%}   "
                f"brush {brush_size}px   hands {len(hands)}   FPS {fps:.0f}"
            )
            cv2.putText(
                display,
                label,
                (12, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if now < status_until:
                cv2.putText(
                    display,
                    status,
                    (12, toolbar.height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (80, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            if show_help:
                draw_help_overlay(display, toolbar.height)

            cv2.imshow("Gesture Canvas", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                status = f"Saved: {save_canvas(canvas, args.output).name}"
                status_until = time.monotonic() + 2
            elif key == ord("c"):
                canvas.clear()
                grab_coordinator.reset()
            elif key == ord("z"):
                canvas.undo()
                grab_coordinator.reset()
            elif key == ord("y"):
                canvas.redo()
                grab_coordinator.reset()
            elif key == ord("["):
                brush_size = adjust_brush_size(brush_size, -1)
                status = f"Brush size: {brush_size}px"
                status_until = time.monotonic() + 2
            elif key == ord("]"):
                brush_size = adjust_brush_size(brush_size, 1)
                status = f"Brush size: {brush_size}px"
                status_until = time.monotonic() + 2
            elif key == ord("h"):
                show_help = not show_help
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
