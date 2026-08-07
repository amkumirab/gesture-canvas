import pytest

from gesture_canvas.toolbar import BUTTONS, DwellSelector, Toolbar


def test_dwell_selects_once_after_required_time():
    selector = DwellSelector(dwell_seconds=0.7)
    button = BUTTONS[0]

    assert selector.update(button, now=10.0) == (None, 0.0)
    selected, progress = selector.update(button, now=10.35)
    assert selected is None
    assert progress == pytest.approx(0.5)
    selected, progress = selector.update(button, now=10.7)
    assert selected == button
    assert progress == 1.0
    assert selector.update(button, now=11.5) == (None, 1.0)


def test_moving_to_another_button_restarts_progress():
    selector = DwellSelector(dwell_seconds=0.7)
    selector.update(BUTTONS[0], now=1.0)
    selector.update(BUTTONS[0], now=1.6)
    assert selector.update(BUTTONS[1], now=1.65) == (None, 0.0)


def test_leaving_toolbar_resets_selection():
    selector = DwellSelector(dwell_seconds=0.7)
    selector.update(BUTTONS[0], now=1.0)
    assert selector.update(None, now=1.5) == (None, 0.0)
    assert selector.update(BUTTONS[0], now=1.6) == (None, 0.0)


def test_toolbar_hit_testing():
    toolbar = Toolbar(frame_width=1280)
    first = toolbar.hit_test((toolbar.margin + 2, toolbar.margin + 2))
    assert first == BUTTONS[0]
    assert toolbar.hit_test((10, toolbar.height + 1)) is None


def test_all_toolbar_buttons_fit_common_camera_width():
    toolbar = Toolbar(frame_width=640)
    last_index = len(BUTTONS) - 1
    last_left = toolbar.margin + last_index * (toolbar.button_width + toolbar.gap)
    assert last_left + toolbar.button_width <= 640 - toolbar.margin


def test_dwell_duration_must_be_positive():
    with pytest.raises(ValueError):
        DwellSelector(dwell_seconds=0)
