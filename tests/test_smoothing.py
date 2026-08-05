import pytest

from gesture_canvas.smoothing import ExponentialSmoother


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

