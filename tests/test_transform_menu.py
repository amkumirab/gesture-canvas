import cv2
import numpy as np
import pytest

from gesture_canvas.transform_menu import MenuHand, TransformControlMenu


def center(rect):
    left, top, right, bottom = rect
    return ((left + right) // 2, (top + bottom) // 2)


def pointing(point, key="left"):
    return MenuHand(key, point, pointing=True)


def open_menu(menu):
    hand = pointing(center(menu.trigger_rect))
    assert menu.update([hand], now=0) is None
    assert menu.blocks_canvas
    assert menu.update([hand], now=0.7) is None
    assert menu.is_open


def test_trigger_opens_once_while_pointer_stays_over_it():
    menu = TransformControlMenu(640, 480)
    open_menu(menu)
    menu.update([pointing(center(menu.trigger_rect))], now=2)
    assert menu.is_open
    assert menu.blocks_canvas


@pytest.mark.parametrize("mode", ["auto", "scale", "tilt_x", "tilt_y", "spin"])
def test_dwell_selects_mode_and_waits_for_neutral_before_resuming(mode):
    menu = TransformControlMenu(640, 480)
    open_menu(menu)
    point = center(menu.option_rects[mode])
    hand = pointing(point)
    assert menu.update([hand], now=1) is None
    assert menu.update([hand], now=1.7) == mode
    assert not menu.is_open
    assert menu.blocks_canvas

    menu.update([hand], now=3)
    assert menu.blocks_canvas
    menu.update([], now=4)
    assert menu.blocks_canvas
    menu.update([MenuHand("left", (100, 200), pinching=True)], now=5)
    assert menu.blocks_canvas
    menu.update([MenuHand("left", (100, 200), erasing=True)], now=6)
    assert menu.blocks_canvas
    menu.update([MenuHand("left", (100, 200))], now=7)
    assert not menu.blocks_canvas


def test_hand_loss_identity_change_and_pinch_restart_dwell():
    menu = TransformControlMenu(640, 480)
    open_menu(menu)
    point = center(menu.option_rects["spin"])
    menu.update([pointing(point)], now=1)
    assert menu.update([pointing(point, "right")], now=1.6) is None
    assert menu.update([MenuHand("right", point, pinching=True)], now=2) is None
    menu.update([pointing(point, "right")], now=2.1)
    menu.update([], now=2.7)
    assert menu.update([pointing(point, "right")], now=2.8) is None
    assert menu.update([pointing(point, "right")], now=3.5) == "spin"


def test_cancel_changes_no_mode_and_requires_neutral():
    menu = TransformControlMenu(640, 480)
    open_menu(menu)
    hand = pointing(center(menu.close_rect))
    assert menu.update([hand], now=1) is None
    assert menu.update([hand], now=1.7) is None
    assert not menu.is_open
    assert menu.blocks_canvas


def test_other_hand_can_interrupt_drawing_without_selecting_a_mode():
    menu = TransformControlMenu(640, 480)
    open_menu(menu)
    menu.update([MenuHand("right", (100, 200), pinching=True)], now=1)
    assert menu.blocks_canvas
    assert menu.is_open


def test_hand_over_menu_can_take_focus_from_a_pointer_elsewhere():
    menu = TransformControlMenu(640, 480)
    point = center(menu.trigger_rect)
    hands = [pointing((500, 200)), pointing(point, key="right")]
    menu.update(hands, now=0)
    menu.update(hands, now=0.7)
    assert menu.is_open


def test_disabled_menu_cancels_hover_and_dismisses_open_panel():
    menu = TransformControlMenu(640, 480)
    open_menu(menu)
    hand = pointing(center(menu.option_rects["tilt_x"]))
    menu.update([hand], now=1)
    assert menu.update([hand], now=2, enabled=False) is None
    assert not menu.is_open
    assert menu.blocks_canvas


@pytest.mark.parametrize("size", [(320, 240), (640, 480), (1280, 720)])
def test_menu_targets_fit_frame_and_selected_mode_is_visible(size, monkeypatch):
    width, height = size
    menu = TransformControlMenu(width, height)
    for rect in [menu.trigger_rect, menu.close_rect, *menu.option_rects.values()]:
        left, top, right, bottom = rect
        assert 0 <= left < right < width
        assert 74 < top < bottom < height
    open_menu(menu)
    labels = []
    original = cv2.putText

    def capture(frame, text, *args, **kwargs):
        labels.append(text)
        return original(frame, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    menu.draw(frame, "tilt_y")
    assert np.any(frame)
    assert any("SELECTED" in label and "TILT Y" in label for label in labels)
