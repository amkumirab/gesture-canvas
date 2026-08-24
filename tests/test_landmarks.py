import pytest

from gesture_canvas.landmarks import Landmark, as_landmarks, to_pixel


def test_media_pipe_points_are_converted_to_landmarks():
    raw = [Landmark(i * 0.01, i * 0.02, i * -0.005) for i in range(21)]
    assert as_landmarks(raw) == raw


def test_to_pixel_clamps_coordinates():
    assert to_pixel(Landmark(-1, 2), 640, 480) == (0, 479)


def test_invalid_landmark_count_is_rejected():
    with pytest.raises(ValueError):
        as_landmarks([Landmark(0, 0)])
