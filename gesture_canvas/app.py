"""Real-time Gesture Canvas application."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from .camera import create_hand_tracker, detect_landmarks, open_camera
from .canvas import DrawingCanvas
from .gestures import (
    DrawGestureStabilizer,
    Gesture,
    PinchDetector,
    RuleBasedGestureClassifier,
)
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
    smoother = AdaptiveSmoother()
    pinch_detector = PinchDetector()
    draw_stabilizer = DrawGestureStabilizer()

    canvas: DrawingCanvas | None = None
    toolbar: Toolbar | None = None
    selector = DwellSelector(dwell_seconds=0.7)
    active_tool = "blue"
    hover_key: str | None = None
    hover_progress = 0.0
    status = "Draw with one finger; pinch any painted shape to move it"
    status_until = time.monotonic() + 4
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

            landmarks = detect_landmarks(tracker, frame)
            gesture = Gesture.IDLE
            confidence = 0.0
            cursor: tuple[int, int] | None = None
            hover_key = None
            hover_progress = 0.0
            pinching = False
            pinch_started = False
            pinch_ended = False
            interaction_time = time.monotonic()

            if landmarks is not None:
                raw_gesture, confidence = classifier.classify(landmarks)
                pinching, pinch_started, pinch_ended, _ = pinch_detector.update(landmarks)
                gesture = draw_stabilizer.update(raw_gesture, interaction_time)
                if pinching:
                    gesture = Gesture.PINCH
                if pinch_ended:
                    draw_blocked_until = interaction_time + 0.22
                cursor = smoother.update(to_pixel(landmarks[8], width, height))
                in_toolbar = cursor[1] <= toolbar.height

                if in_toolbar:
                    canvas.end_stroke()
                    canvas.end_move()
                    can_select = gesture is Gesture.DRAW and not pinching
                    hovered = toolbar.hit_test(cursor) if can_select else None
                    activated, hover_progress = selector.update(
                        hovered, time.monotonic()
                    )
                    hover_key = hovered.key if hovered else None
                    if activated:
                        active_tool, status = apply_button(
                            activated, canvas, args.output, active_tool
                        )
                        status_until = time.monotonic() + 2
                else:
                    selector.reset()
                    drawing = (
                        gesture is Gesture.DRAW
                        and interaction_time >= draw_blocked_until
                    )
                    two_finger_erase = gesture is Gesture.ERASE
                    if pinching:
                        canvas.end_stroke()
                        if pinch_started:
                            canvas.begin_move(cursor, selection_radius=32)
                        if canvas.is_moving:
                            canvas.update_move(cursor)
                    elif drawing:
                        canvas.end_move()
                        if active_tool == "eraser":
                            canvas.erase_point(cursor, 46, max_segment_length=110)
                        else:
                            canvas.add_point(
                                cursor,
                                COLORS[active_tool],
                                7,
                                max_segment_length=90,
                            )
                    elif two_finger_erase:
                        canvas.end_move()
                        canvas.erase_point(cursor, 46, max_segment_length=110)
                    else:
                        canvas.end_stroke()
                        canvas.end_move()
            else:
                if not draw_stabilizer.hold_during_missing(interaction_time):
                    canvas.end_stroke()
                canvas.end_move()
                if pinch_detector.active:
                    draw_blocked_until = interaction_time + 0.22
                pinch_detector.reset()
                smoother.reset()
                selector.reset()

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
            toolbar.draw(display, active_tool, hover_key, hover_progress)
            if cursor is not None:
                radius = 22 if gesture is Gesture.ERASE or active_tool == "eraser" else 9
                if pinching:
                    radius = 14
                cursor_color = (255, 255, 255) if hover_key else (80, 220, 255)
                cv2.circle(display, cursor, radius, cursor_color, 2, cv2.LINE_AA)

            now = time.monotonic()
            elapsed = max(now - previous_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
            previous_time = now
            label = f"{gesture.value}  {confidence:.0%}   FPS {fps:.0f}"
            cv2.putText(display, label, (12, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            if now < status_until:
                cv2.putText(display, status, (12, toolbar.height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow("Gesture Canvas", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                status = f"Saved: {save_canvas(canvas, args.output).name}"
                status_until = time.monotonic() + 2
            elif key == ord("c"):
                canvas.clear()
            elif key == ord("z"):
                canvas.undo()
            elif key == ord("y"):
                canvas.redo()
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
