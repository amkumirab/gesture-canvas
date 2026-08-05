"""Train and evaluate the optional PyTorch gesture MLP."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .model import build_mlp, require_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a personal gesture classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/gestures.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/gesture_mlp.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    features: list[list[float]] = []
    labels: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels.append(row["label"])
            features.append([float(row[f"f{i}"]) for i in range(63)])
    if not features:
        raise ValueError("Dataset is empty")
    return np.asarray(features, dtype=np.float32), labels


def stratified_indices(labels: list[str], seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    train: list[int] = []
    test: list[int] = []
    for group in groups.values():
        rng.shuffle(group)
        split = max(1, int(len(group) * 0.8))
        if split == len(group) and len(group) > 1:
            split -= 1
        train.extend(group[:split])
        test.extend(group[split:])
    if not test:
        raise ValueError("Collect at least two samples for every label")
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main() -> None:
    args = parse_args()
    torch, nn = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, raw_labels = load_dataset(args.data)
    counts = Counter(raw_labels)
    if len(counts) < 2:
        raise ValueError("Collect at least two different gesture labels")
    labels = sorted(counts)
    label_to_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([label_to_index[label] for label in raw_labels], dtype=np.int64)
    train_idx, test_idx = stratified_indices(raw_labels, args.seed)

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_test = torch.from_numpy(x[test_idx])
    y_test = torch.from_numpy(y[test_idx])

    model = build_mlp(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train))
        running_loss = 0.0
        for start in range(0, len(x_train), 64):
            batch = permutation[start : start + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} loss={running_loss / len(x_train):.4f}")

    model.eval()
    with torch.inference_mode():
        predictions = model(x_test).argmax(dim=1)
    accuracy = float((predictions == y_test).float().mean().item())
    print(f"test_accuracy={accuracy:.3f} test_samples={len(test_idx)}")
    for label, index in label_to_index.items():
        mask = y_test == index
        if mask.any():
            class_accuracy = float((predictions[mask] == y_test[mask]).float().mean().item())
            print(f"  {label}: accuracy={class_accuracy:.3f} samples={int(mask.sum())}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "labels": labels, "feature_version": 1},
        args.output,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()

