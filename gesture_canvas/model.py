"""Optional PyTorch MLP used for user-trained gesture classification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .gestures import Gesture
from .landmarks import Landmark, normalized_features


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for learned gestures. Install with: pip install -e .[ml]"
        ) from exc
    return torch, nn


def build_mlp(class_count: int):
    torch, nn = require_torch()
    del torch
    return nn.Sequential(
        nn.Linear(63, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, class_count),
    )


class NeuralGesturePredictor:
    def __init__(self, checkpoint_path: Path, confidence_threshold: float = 0.72) -> None:
        torch, _ = require_torch()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.labels: list[str] = checkpoint["labels"]
        self.model = build_mlp(len(self.labels))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = confidence_threshold
        self.torch = torch

    def classify(self, landmarks: Sequence[Landmark]) -> tuple[Gesture, float]:
        features = normalized_features(landmarks)
        tensor = self.torch.from_numpy(features).unsqueeze(0)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(tensor), dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        score = float(confidence.item())
        if score < self.threshold:
            return Gesture.IDLE, score
        try:
            return Gesture(self.labels[int(index.item())]), score
        except ValueError:
            return Gesture.IDLE, score

