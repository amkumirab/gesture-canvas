import numpy as np
import pytest

from gesture_canvas.landmarks import Landmark, normalized_features, to_pixel


def test_features_are_translation_and_scale_invariant():
    base = [Landmark(i * 0.01, i * 0.02, i * -0.005) for i in range(21)]
    moved = [Landmark(p.x * 2 + 0.4, p.y * 2 - 0.2, p.z * 2 + 0.1) for p in base]
    np.testing.assert_allclose(normalized_features(base), normalized_features(moved), atol=1e-5)


def test_feature_vector_has_expected_size():
    hand = [Landmark(i * 0.01, i * 0.01, 0) for i in range(21)]
    assert normalized_features(hand).shape == (63,)


def test_to_pixel_clamps_coordinates():
    assert to_pixel(Landmark(-1, 2), 640, 480) == (0, 479)


def test_invalid_landmark_count_is_rejected():
    with pytest.raises(ValueError):
        normalized_features([Landmark(0, 0)])

