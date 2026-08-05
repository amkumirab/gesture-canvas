import csv
from pathlib import Path

from gesture_canvas.train import load_dataset, stratified_indices


def test_dataset_loading_and_stratified_split(tmp_path: Path):
    path = tmp_path / "gestures.csv"
    fields = ["label", *[f"f{i}" for i in range(63)]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in ("draw", "draw", "erase", "erase"):
            row = {"label": label}
            row.update({f"f{i}": i / 100 for i in range(63)})
            writer.writerow(row)

    features, labels = load_dataset(path)
    train, test = stratified_indices(labels, seed=42)
    assert features.shape == (4, 63)
    assert len(train) == 2
    assert len(test) == 2
    assert {labels[i] for i in test} == {"draw", "erase"}

