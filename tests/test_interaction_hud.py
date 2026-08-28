import cv2
import numpy as np
import pytest

from gesture_canvas.interaction_hud import draw_manipulation_hud


@pytest.mark.parametrize("mode", ["move", "move-3d", "transform"])
def test_manipulation_hud_draws_for_active_modes(mode):
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    draw_manipulation_hud(frame, mode, (1.2, 10, -15, 25))
    assert np.count_nonzero(frame) > 0


def test_manipulation_hud_stays_hidden_while_idle():
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    draw_manipulation_hud(frame, "idle")
    assert not np.any(frame)


def test_manipulation_hud_rejects_unknown_mode():
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        draw_manipulation_hud(frame, "broken")
