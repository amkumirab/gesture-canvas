import pytest

from gesture_canvas.landmarks import (
    Landmark,
    as_landmarks,
    palm_span,
    pinch_point,
    to_pixel,
)


def test_media_pipe_points_are_converted_to_landmarks():
    raw = [Landmark(i * 0.01, i * 0.02, i * -0.005) for i in range(21)]
    assert as_landmarks(raw) == raw


def test_to_pixel_clamps_coordinates():
    assert to_pixel(Landmark(-1, 2), 640, 480) == (0, 479)


def test_palm_span_uses_index_to_pinky_base_distance():
    hand = [Landmark(0.5, 0.5) for _ in range(21)]
    hand[5] = Landmark(0.2, 0.3)
    hand[17] = Landmark(0.5, 0.7)
    assert palm_span(hand) == pytest.approx(0.5)


def test_pinch_point_is_midway_between_thumb_and_index():
    hand = [Landmark(0.5, 0.5) for _ in range(21)]
    hand[4] = Landmark(0.2, 0.4, -0.1)
    hand[8] = Landmark(0.6, 0.8, -0.3)
    point = pinch_point(hand)
    assert (point.x, point.y, point.z) == pytest.approx((0.4, 0.6, -0.2))


def test_invalid_landmark_count_is_rejected():
    with pytest.raises(ValueError):
        as_landmarks([Landmark(0, 0)])
    with pytest.raises(ValueError):
        palm_span([Landmark(0, 0)])
    with pytest.raises(ValueError):
        pinch_point([Landmark(0, 0)])
