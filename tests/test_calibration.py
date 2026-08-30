import pytest

from gesture_canvas.calibration import (
    CAPTURE_OPEN,
    CAPTURE_PINCH,
    COMPLETE,
    FAILED,
    PREPARE_PINCH,
    CalibrationMeasurement,
    CalibrationSession,
    light_quality,
)


def measurement(ratio, span=0.20, cursor=(0.5, 0.5)):
    return CalibrationMeasurement(ratio, span, cursor)


def complete_phase(session, ratio, start_time):
    result = None
    for index in range(session.samples_per_phase):
        result = session.update(
            measurement(ratio, cursor=(0.5 + index * 0.001, 0.5)),
            start_time + index * 0.01,
        )
    return result


def test_calibration_derives_separated_thresholds_and_smoothing():
    session = CalibrationSession(
        now=0,
        camera_index=1,
        samples_per_phase=4,
        preparation_seconds=1,
    )
    assert session.update(None, now=1) is None
    assert session.stage == CAPTURE_OPEN
    complete_phase(session, ratio=1.05, start_time=1.1)
    assert session.stage == PREPARE_PINCH

    session.update(None, now=2.2)
    assert session.stage == CAPTURE_PINCH
    settings = complete_phase(session, ratio=0.12, start_time=2.3)

    assert session.stage == COMPLETE
    assert settings.calibrated
    assert settings.camera_index == 1
    assert 0.12 < settings.pinch_close_ratio < settings.pinch_release_ratio < 1
    assert settings.draw_min_alpha < settings.draw_max_alpha
    assert settings.selection_radius_ratio > 0


def test_calibration_rejects_poses_that_are_too_similar():
    session = CalibrationSession(
        now=0,
        samples_per_phase=3,
        preparation_seconds=0,
    )
    session.update(None, now=0)
    complete_phase(session, ratio=0.40, start_time=0.1)
    session.update(None, now=0.2)
    assert complete_phase(session, ratio=0.30, start_time=0.3) is None
    assert session.stage == FAILED
    assert "TOO SIMILAR" in session.instruction


def test_calibration_rejects_an_incomplete_pinch():
    session = CalibrationSession(
        now=0,
        samples_per_phase=3,
        preparation_seconds=0,
    )
    session.update(None, now=0)
    complete_phase(session, ratio=0.80, start_time=0.1)
    session.update(None, now=0.2)
    assert complete_phase(session, ratio=0.50, start_time=0.3) is None
    assert session.stage == FAILED
    assert "NOT CLOSED ENOUGH" in session.instruction


def test_calibration_waits_for_a_visible_hand_and_reports_progress():
    session = CalibrationSession(
        now=0,
        samples_per_phase=4,
        preparation_seconds=0,
    )
    session.update(None, now=0)
    assert session.update(None, now=0.1) is None
    assert session.progress == 0
    session.update(measurement(1.0), now=0.2)
    assert session.progress == 0.25


def test_calibration_validates_inputs_and_light_quality():
    with pytest.raises(ValueError):
        CalibrationSession(now=0, samples_per_phase=2)
    with pytest.raises(ValueError):
        CalibrationSession(now=0, preparation_seconds=-1)
    with pytest.raises(ValueError):
        CalibrationMeasurement(0, 0.2, (0.5, 0.5))

    assert light_quality(20) == "LOW LIGHT"
    assert light_quality(120) == "LIGHT OK"
    assert light_quality(240) == "TOO BRIGHT"
    with pytest.raises(ValueError):
        light_quality(float("nan"))
    with pytest.raises(ValueError):
        light_quality(300)
