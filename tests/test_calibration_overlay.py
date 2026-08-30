import numpy as np

from gesture_canvas.calibration import CalibrationSession
from gesture_canvas.calibration_overlay import draw_calibration_overlay


def test_calibration_overlay_draws_card_and_progress():
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    session = CalibrationSession(now=0, samples_per_phase=3, preparation_seconds=0)
    session.update(None, now=0)

    draw_calibration_overlay(
        frame,
        session,
        brightness=120,
        hand_confidence=0.9,
        camera_index=0,
    )
    assert np.count_nonzero(frame) > 0


def test_calibration_overlay_skips_tiny_frames():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    session = CalibrationSession(now=0)
    draw_calibration_overlay(frame, session, 120, 0.9, 0)
    assert not np.any(frame)
