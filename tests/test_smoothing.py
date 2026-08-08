import pytest

from gesture_canvas.smoothing import AdaptiveSmoother, ExponentialSmoother


def test_smoothing_reduces_a_large_jump():
    smoother = ExponentialSmoother(alpha=0.5)
    assert smoother.update((0, 0)) == (0, 0)
    assert smoother.update((100, 100)) == (50, 50)


def test_reset_forgets_previous_position():
    smoother = ExponentialSmoother(alpha=0.2)
    smoother.update((0, 0))
    smoother.reset()
    assert smoother.update((100, 50)) == (100, 50)


def test_alpha_validation():
    with pytest.raises(ValueError):
        ExponentialSmoother(alpha=0)


def test_adaptive_smoother_filters_small_jitter():
    smoother = AdaptiveSmoother(min_alpha=0.2, max_alpha=0.8, response_distance=100)
    assert smoother.update((100, 100)) == (100, 100)
    assert smoother.update((110, 100))[0] < 105


def test_adaptive_smoother_responds_to_large_motion():
    smoother = AdaptiveSmoother(min_alpha=0.2, max_alpha=0.8, response_distance=100)
    smoother.update((0, 0))
    x, _ = smoother.update((100, 0))
    assert x == 80


def test_adaptive_smoother_validation():
    with pytest.raises(ValueError):
        AdaptiveSmoother(min_alpha=0.8, max_alpha=0.2)
    with pytest.raises(ValueError):
        AdaptiveSmoother(response_distance=0)
