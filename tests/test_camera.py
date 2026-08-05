from pathlib import Path

from gesture_canvas.camera import ensure_hand_model


def test_existing_model_is_not_downloaded(tmp_path: Path, monkeypatch):
    model = tmp_path / "hand.task"
    model.write_bytes(b"existing")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr("urllib.request.urlretrieve", fail_if_called)
    assert ensure_hand_model(model) == model


def test_model_download_is_atomically_moved(tmp_path: Path, monkeypatch):
    model = tmp_path / "nested" / "hand.task"

    def fake_download(_url, target):
        Path(target).write_bytes(b"model-data")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_download)
    assert ensure_hand_model(model) == model
    assert model.read_bytes() == b"model-data"
    assert not model.with_suffix(".download").exists()

