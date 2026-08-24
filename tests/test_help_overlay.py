import cv2
import numpy as np

from gesture_canvas.help_overlay import HELP_ITEMS, draw_help_overlay


def test_help_overlay_draws_a_visible_panel():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    draw_help_overlay(frame)

    assert np.count_nonzero(frame) > 0
    assert tuple(frame[86, 21]) == (80, 220, 255)


def test_help_overlay_skips_frames_that_are_too_small(monkeypatch):
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    rectangle_calls = 0

    def count_rectangle(*args, **kwargs):
        nonlocal rectangle_calls
        rectangle_calls += 1

    monkeypatch.setattr(cv2, "rectangle", count_rectangle)

    draw_help_overlay(frame)

    assert rectangle_calls == 0
    assert len(HELP_ITEMS) >= 9
