from pathlib import Path

import pytest

from gesture_canvas.gestures import Gesture
from gesture_canvas.landmarks import Landmark
from gesture_canvas.model import NeuralGesturePredictor, build_mlp


torch = pytest.importorskip("torch")


def test_checkpoint_round_trip(tmp_path: Path):
    model = build_mlp(class_count=2)
    checkpoint = tmp_path / "gesture.pt"
    torch.save(
        {"state_dict": model.state_dict(), "labels": ["draw", "erase"], "feature_version": 1},
        checkpoint,
    )
    predictor = NeuralGesturePredictor(checkpoint, confidence_threshold=0.0)
    hand = [Landmark(i * 0.01, i * 0.02, i * -0.002) for i in range(21)]
    gesture, confidence = predictor.classify(hand)
    assert gesture in {Gesture.DRAW, Gesture.ERASE}
    assert 0 <= confidence <= 1

