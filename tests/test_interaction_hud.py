import cv2
import numpy as np
import pytest

from gesture_canvas.interaction_hud import draw_manipulation_hud, draw_transform_mode_hud


@pytest.mark.parametrize(
    "mode",
    ["move", "move-3d", "arm-transform", "transform"],
)
def test_manipulation_hud_draws_for_active_modes(mode):
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    draw_manipulation_hud(frame, mode, (1.2, 10, -15, 25))
    assert np.count_nonzero(frame) > 0


def test_transform_hud_draws_locked_control_feedback():
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    draw_manipulation_hud(
        frame,
        "transform",
        (1.2, 10, -15, 25),
        active_control="spin",
    )
    assert np.count_nonzero(frame) > 0


def test_manipulation_hud_stays_hidden_while_idle():
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    draw_manipulation_hud(frame, "idle")
    assert not np.any(frame)


def test_manipulation_hud_rejects_unknown_mode():
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        draw_manipulation_hud(frame, "broken")


@pytest.mark.parametrize("control", ["auto", "scale", "tilt_x", "tilt_y", "spin"])
def test_mode_hud_shows_selected_control_without_active_gesture(control, monkeypatch):
    labels = []
    original = cv2.putText

    def capture_label(frame, text, *args, **kwargs):
        labels.append(text)
        return original(frame, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_label)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_transform_mode_hud(frame, control)
    assert np.any(frame)
    assert any("0 AUTO" in text and "4 Z" in text for text in labels)


def test_manual_transform_hint_explains_key_switching(monkeypatch):
    labels = []
    original = cv2.putText

    def capture_label(frame, text, *args, **kwargs):
        labels.append(text)
        return original(frame, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_label)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_manipulation_hud(
        frame,
        "transform",
        active_control="tilt_y",
        requested_control="tilt_y",
    )
    assert any("MANUAL" in text and "0-4" in text for text in labels)
