"""Real-time Gesture Canvas application."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from pathlib import Path

import cv2

from .camera import DetectedHand, create_hand_tracker, detect_hands, open_camera
from .calibration import CalibrationMeasurement, CalibrationSession
from .calibration_overlay import draw_calibration_overlay
from .canvas import DrawingCanvas
from .controls import DEFAULT_BRUSH_SIZE, adjust_brush_size
from .gestures import (
    DrawGestureStabilizer,
    Gesture,
    GestureRecognizer,
    PinchDetector,
)
from .help_overlay import draw_help_overlay
from .interaction_hud import draw_manipulation_hud
from .interaction import GrabCoordinator, PinchHand
from .landmarks import palm_span, pinch_point, to_pixel
from .project_file import ProjectFormatError, load_project, save_project
from .smoothing import AdaptiveSmoother, ScalarSmoother
from .settings import (
    GestureSettings,
    default_settings_path,
    load_settings,
    save_settings,
)
from .spatial_guides import draw_spatial_guides
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
    depth_signal: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw in the air using hand gestures.")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Webcam index (default: saved setting or 0)",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Saved image folder")
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Custom settings file path",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Editable .gcanvas project to open at startup",
    )
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_canvas(canvas: DrawingCanvas, output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return canvas.export(output_dir / f"gesture_canvas_{stamp}.png")


def choose_project_file(save: bool, current: Path | None = None) -> Path | None:
    """Show the operating system project picker and return the chosen path."""

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        options = {
            "title": "Save Gesture Canvas project" if save else "Open Gesture Canvas project",
            "filetypes": (("Gesture Canvas project", "*.gcanvas"), ("All files", "*.*")),
        }
        if current is not None:
            options["initialdir"] = str(current.parent)
            options["initialfile"] = current.name
        elif save:
            options["initialfile"] = (
                f"gesture_canvas_{datetime.now():%Y%m%d_%H%M%S}.gcanvas"
            )
        try:
            selected = (
                filedialog.asksaveasfilename(
                    **options,
                    defaultextension=".gcanvas",
                )
                if save
                else filedialog.askopenfilename(**options)
            )
        finally:
            root.destroy()
    except Exception as error:
        raise OSError("The project file picker could not be opened") from error
    return Path(selected) if selected else None


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


def open_selected_camera(
    requested_index: int | None,
    saved_index: int,
):
    """Open an explicit camera, or safely fall back from a stale saved camera."""

    camera_index = saved_index if requested_index is None else requested_index
    if camera_index < 0:
        raise ValueError("Camera index cannot be negative")
    try:
        return open_camera(camera_index), camera_index
    except RuntimeError:
        if requested_index is not None or camera_index == 0:
            raise
        return open_camera(0), 0


def main() -> None:
    args = parse_args()
    settings_path = args.settings or default_settings_path()
    settings = load_settings(settings_path)
    recognizer = GestureRecognizer(settings.pinch_close_ratio)
    camera, camera_index = open_selected_camera(
        args.camera,
        settings.camera_index,
    )
    if camera_index != settings.camera_index:
        settings = replace(settings, camera_index=camera_index)
        try:
            save_settings(settings, settings_path)
        except OSError:
            pass
    try:
        tracker = create_hand_tracker()
    except Exception:
        camera.release()
        raise

    pinch_detectors: dict[str, PinchDetector] = {}
    draw_smoothers: dict[str, AdaptiveSmoother] = {}
    pinch_smoothers: dict[str, AdaptiveSmoother] = {}
    depth_smoothers: dict[str, ScalarSmoother] = {}
    draw_stabilizers: dict[str, DrawGestureStabilizer] = {}
    missing_hand_frames: dict[str, int] = {}
    grab_coordinator = GrabCoordinator(selection_radius=settings.selection_radius(1280))

    canvas: DrawingCanvas | None = None
    toolbar: Toolbar | None = None
    selector = DwellSelector(dwell_seconds=0.7)
    active_tool = "blue"
    brush_size = DEFAULT_BRUSH_SIZE
    show_help = False
    drawing_hand_id: str | None = None
    hover_key: str | None = None
    hover_progress = 0.0
    calibration: CalibrationSession | None = None
    current_project: Path | None = None
    pending_project = args.project
    status = (
        "Calibration loaded - press K to recalibrate"
        if settings.calibrated
        else "Press K to calibrate gestures, or start with the defaults"
    )
    status_until = time.monotonic() + 5
    draw_blocked_until = 0.0
    previous_time = time.monotonic()
    fps = 0.0
    failed_camera_reads = 0

    def apply_interaction_settings() -> None:
        nonlocal recognizer
        recognizer = GestureRecognizer(settings.pinch_close_ratio)
        pinch_detectors.clear()
        draw_smoothers.clear()
        pinch_smoothers.clear()
        depth_smoothers.clear()
        draw_stabilizers.clear()
        missing_hand_frames.clear()
        grab_coordinator.reset(canvas)
        frame_width = canvas.width if canvas is not None else 1280
        grab_coordinator.selection_radius = settings.selection_radius(frame_width)
        if canvas is not None:
            canvas.spatial.depth_sensitivity = settings.depth_sensitivity

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                failed_camera_reads += 1
                if failed_camera_reads <= 8:
                    time.sleep(0.02)
                    continue
                raise RuntimeError("The webcam stopped returning frames")
            failed_camera_reads = 0
            if args.mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            if canvas is None:
                canvas = DrawingCanvas(
                    width,
                    height,
                    depth_sensitivity=settings.depth_sensitivity,
                )
                toolbar = Toolbar(width)
                grab_coordinator.selection_radius = settings.selection_radius(width)
                if pending_project is not None:
                    try:
                        opened = load_project(
                            pending_project,
                            target_size=(canvas.width, canvas.height),
                        )
                        canvas.restore_project_state(opened.canvas)
                        brush_size = opened.brush_size
                        active_tool = opened.active_tool
                        current_project = pending_project.resolve()
                        status = f"Project opened: {pending_project.name}"
                    except ProjectFormatError as error:
                        status = f"Could not open project: {error}"
                    status_until = time.monotonic() + 4
                    pending_project = None
            assert toolbar is not None

            interaction_time = time.monotonic()
            brightness = float(frame.mean())
            detected = detect_hands(tracker, frame)
            hands: list[HandFrame] = []
            calibration_measurements: list[
                tuple[float, CalibrationMeasurement]
            ] = []
            used_keys: set[str] = set()
            seen_keys: set[str] = set()

            for index, detected_hand in enumerate(detected):
                key = hand_key(detected_hand, index, used_keys)
                seen_keys.add(key)
                missing_hand_frames.pop(key, None)
                pinch_detector = pinch_detectors.setdefault(
                    key,
                    PinchDetector(
                        close_ratio=settings.pinch_close_ratio,
                        release_ratio=settings.pinch_release_ratio,
                    ),
                )
                draw_smoother = draw_smoothers.setdefault(
                    key,
                    AdaptiveSmoother(
                        min_alpha=settings.draw_min_alpha,
                        max_alpha=settings.draw_max_alpha,
                    ),
                )
                pinch_smoother = pinch_smoothers.setdefault(
                    key,
                    AdaptiveSmoother(
                        min_alpha=settings.pinch_min_alpha,
                        max_alpha=settings.pinch_max_alpha,
                        response_distance=55.0,
                    ),
                )
                depth_smoother = depth_smoothers.setdefault(
                    key,
                    ScalarSmoother(alpha=settings.depth_alpha),
                )
                stabilizer = draw_stabilizers.setdefault(key, DrawGestureStabilizer())
                raw_gesture, confidence = recognizer.recognize(
                    detected_hand.landmarks
                )
                pinching, pinch_started, pinch_ended, ratio = (
                    pinch_detector.update(detected_hand.landmarks)
                )
                gesture = stabilizer.update(raw_gesture, interaction_time)
                if pinching:
                    gesture = Gesture.PINCH
                if pinch_ended:
                    draw_blocked_until = interaction_time + 0.22
                draw_cursor = draw_smoother.update(
                    to_pixel(detected_hand.landmarks[8], width, height)
                )
                pinch_cursor = pinch_smoother.update(
                    to_pixel(pinch_point(detected_hand.landmarks), width, height)
                )
                cursor = pinch_cursor if pinching else draw_cursor
                raw_palm_span = palm_span(detected_hand.landmarks)
                depth_signal = depth_smoother.update(raw_palm_span)
                if isfinite(ratio) and ratio > 0 and raw_palm_span > 0:
                    index_tip = detected_hand.landmarks[8]
                    calibration_measurements.append(
                        (
                            detected_hand.confidence,
                            CalibrationMeasurement(
                                ratio,
                                raw_palm_span,
                                (index_tip.x, index_tip.y),
                            ),
                        )
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
                        depth_signal=depth_signal,
                    )
                )

            for key, detector in pinch_detectors.items():
                if key not in seen_keys:
                    missing = missing_hand_frames.get(key, 0) + 1
                    missing_hand_frames[key] = missing
                    if missing > grab_coordinator.missing_grace_frames:
                        if detector.active:
                            draw_blocked_until = interaction_time + 0.22
                        detector.reset()
                        draw_smoothers[key].reset()
                        pinch_smoothers[key].reset()
                        depth_smoothers[key].reset()

            best_hand_confidence = max(
                (confidence for confidence, _ in calibration_measurements),
                default=0.0,
            )
            calibration_frame = calibration is not None
            if calibration is not None:
                measurement = (
                    max(calibration_measurements, key=lambda item: item[0])[1]
                    if calibration_measurements
                    else None
                )
                calibrated_settings = calibration.update(
                    measurement,
                    interaction_time,
                )
                if calibrated_settings is not None:
                    settings = calibrated_settings
                    saved = True
                    try:
                        save_settings(settings, settings_path)
                    except OSError:
                        saved = False
                    calibration = None
                    apply_interaction_settings()
                    status = (
                        "Calibration saved"
                        if saved
                        else "Calibration applied; settings could not be saved"
                    )
                    status_until = interaction_time + 3

            pinch_hands = [
                PinchHand(
                    key=hand.key,
                    cursor=hand.cursor,
                    pinching=hand.pinching,
                    started=hand.pinch_started,
                    depth_signal=hand.depth_signal,
                )
                for hand in hands
            ]
            was_moving = canvas.is_moving
            if calibration_frame:
                grab_coordinator.reset(canvas)
                manipulation_frame = False
            else:
                grab_coordinator.update(
                    canvas,
                    pinch_hands,
                    canvas_top=toolbar.height,
                )
                manipulation_frame = (
                    was_moving
                    or canvas.is_moving
                    or any(hand.pinching for hand in hands)
                )

            hover_key = None
            hover_progress = 0.0
            display_gesture = Gesture.PINCH if manipulation_frame else Gesture.IDLE
            display_confidence = 0.0

            if calibration_frame:
                canvas.end_stroke()
                selector.reset()
            elif manipulation_frame:
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
                                    max_segment_length=150,
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
                if canvas.selected_is_spatial and canvas.spatial_transform_info:
                    scale, rotation_x, rotation_y, rotation_z = (
                        canvas.spatial_transform_info
                    )
                    z_position = canvas.selected_spatial_z or 0
                    cv2.putText(
                        display,
                        (
                            f"3D {scale:.2f}x  tilt {rotation_x:+.0f}/"
                            f"{rotation_y:+.0f}  spin {rotation_z:+.0f}  z {z_position:+.0f}"
                        ),
                        (max(8, x), max(toolbar.height + 24, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (80, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
                elif canvas.is_transforming and canvas.transform_info:
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

            spatial_control = canvas.spatial_transform_control
            if canvas.selected_is_spatial:
                draw_spatial_guides(
                    display,
                    canvas.spatial_guide,
                    toolbar.height,
                    active_control=spatial_control,
                )

            if grab_coordinator.arming_transform:
                manipulation_mode = "arm-transform"
            elif canvas.is_transforming:
                manipulation_mode = "transform"
            elif canvas.is_moving:
                manipulation_mode = (
                    "move-3d" if canvas.selected_is_spatial else "move"
                )
            else:
                manipulation_mode = "idle"
            draw_manipulation_hud(
                display,
                manipulation_mode,
                canvas.spatial_transform_info if canvas.selected_is_spatial else None,
                grab_coordinator.recovering_tracking,
                toolbar.height,
                active_control=spatial_control,
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
                f"brush {brush_size}px   3D {canvas.spatial_object_count}   "
                f"hands {len(hands)}   FPS {fps:.0f}"
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
            if calibration is not None:
                draw_calibration_overlay(
                    display,
                    calibration,
                    brightness,
                    best_hand_confidence,
                    camera_index,
                )
            elif show_help:
                draw_help_overlay(display, toolbar.height)

            cv2.imshow("Gesture Canvas", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if calibration is not None:
                if key == 27:
                    calibration = None
                    apply_interaction_settings()
                    status = "Calibration cancelled"
                    status_until = time.monotonic() + 2
                elif key in (ord("k"), ord("K")):
                    calibration = CalibrationSession(
                        time.monotonic(),
                        camera_index=camera_index,
                    )
                elif key in (ord("d"), ord("D")):
                    settings = GestureSettings(camera_index=camera_index)
                    try:
                        save_settings(settings, settings_path)
                        status = "Default gesture settings restored"
                    except OSError:
                        status = "Defaults applied; settings could not be saved"
                    calibration = None
                    apply_interaction_settings()
                    status_until = time.monotonic() + 3
                continue
            if key == 27:
                break
            if key == 19:  # Ctrl+S
                grab_coordinator.reset(canvas)
                selector.reset()
                try:
                    selected = current_project or choose_project_file(True)
                    if selected is None:
                        status = "Project save cancelled"
                    else:
                        saved_project = save_project(
                            selected,
                            canvas.project_state(),
                            brush_size,
                            active_tool,
                        )
                        current_project = saved_project.resolve()
                        status = f"Project saved: {current_project.name}"
                except (OSError, ValueError) as error:
                    status = f"Could not save project: {error}"
                drawing_hand_id = None
                apply_interaction_settings()
                status_until = time.monotonic() + 3
            elif key == 15:  # Ctrl+O
                grab_coordinator.reset(canvas)
                selector.reset()
                try:
                    selected = choose_project_file(False, current_project)
                    if selected is None:
                        status = "Open project cancelled"
                    else:
                        opened = load_project(
                            selected,
                            target_size=(canvas.width, canvas.height),
                        )
                        canvas.restore_project_state(opened.canvas)
                        brush_size = opened.brush_size
                        active_tool = opened.active_tool
                        current_project = selected.resolve()
                        status = f"Project opened: {selected.name}"
                        drawing_hand_id = None
                        apply_interaction_settings()
                except (OSError, ProjectFormatError, ValueError) as error:
                    status = f"Could not open project: {error}"
                status_until = time.monotonic() + 4
            elif key == ord("s"):
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
            elif key in (ord("k"), ord("K")):
                grab_coordinator.reset(canvas)
                canvas.end_stroke()
                selector.reset()
                calibration = CalibrationSession(
                    time.monotonic(),
                    camera_index=camera_index,
                )
            elif key in (ord("d"), ord("D")):
                settings = GestureSettings(camera_index=camera_index)
                try:
                    save_settings(settings, settings_path)
                    status = "Default gesture settings restored"
                except OSError:
                    status = "Defaults applied; settings could not be saved"
                apply_interaction_settings()
                status_until = time.monotonic() + 3
            elif key == ord("r"):
                grab_coordinator.reset(canvas)
                if canvas.reset_spatial_rotation():
                    status = "3D rotation reset"
                else:
                    status = "Select a rotated 3D shape first"
                status_until = time.monotonic() + 2
            elif key == ord("e"):
                cursor_hand = hands_by_key.get(drawing_hand_id or "")
                if cursor_hand is None and hands:
                    cursor_hand = max(hands, key=lambda hand: hand.confidence)
                if cursor_hand is None:
                    status = "Show a hand and point at a closed shape first"
                else:
                    grab_coordinator.reset(canvas)
                    if canvas.promote_to_3d(
                        cursor_hand.cursor,
                        selection_radius=settings.selection_radius(width),
                    ):
                        status = "3D layer created; pinch and move closer or farther"
                    else:
                        status = "Point inside a filled closed shape and press E"
                status_until = time.monotonic() + 3
            elif key in (ord("-"), ord("_")):
                if canvas.adjust_spatial_z(-45):
                    status = f"3D Z: {canvas.selected_spatial_z:+.0f}"
                else:
                    status = "Select or create a 3D shape first"
                status_until = time.monotonic() + 2
            elif key in (ord("="), ord("+")):
                if canvas.adjust_spatial_z(45):
                    status = f"3D Z: {canvas.selected_spatial_z:+.0f}"
                else:
                    status = "Select or create a 3D shape first"
                status_until = time.monotonic() + 2
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
