from pathlib import Path

from gesture_canvas.camera import ensure_landmarker_asset


def test_existing_asset_is_not_downloaded(tmp_path: Path, monkeypatch):
    asset = tmp_path / "hand.task"
    asset.write_bytes(b"existing")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr("urllib.request.urlretrieve", fail_if_called)
    assert ensure_landmarker_asset(asset) == asset


def test_asset_download_is_atomically_moved(tmp_path: Path, monkeypatch):
    asset = tmp_path / "nested" / "hand.task"

    def fake_download(_url, target):
        Path(target).write_bytes(b"asset-data")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_download)
    assert ensure_landmarker_asset(asset) == asset
    assert asset.read_bytes() == b"asset-data"
    assert not asset.with_suffix(".download").exists()
