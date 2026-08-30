import json

import pytest

from gesture_canvas.settings import (
    GestureSettings,
    default_settings_path,
    load_settings,
    save_settings,
)


def test_settings_round_trip(tmp_path):
    path = tmp_path / "nested" / "settings.json"
    settings = GestureSettings(
        camera_index=1,
        pinch_close_ratio=0.28,
        pinch_release_ratio=0.48,
        calibrated=True,
    )

    assert save_settings(settings, path) == path
    assert load_settings(path) == settings
    assert not path.with_suffix(".json.tmp").exists()


def test_invalid_or_unknown_settings_fall_back_safely(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("not json", encoding="utf-8")
    assert load_settings(path) == GestureSettings()

    path.write_text(json.dumps({"pinch_close_ratio": 0.8}), encoding="utf-8")
    assert load_settings(path) == GestureSettings()


def test_unknown_json_fields_are_ignored():
    settings = GestureSettings.from_dict({"future_option": 123})
    assert settings == GestureSettings()


def test_settings_validate_ranges_and_selection_radius():
    with pytest.raises(ValueError):
        GestureSettings(camera_index=-1)
    with pytest.raises(ValueError):
        GestureSettings(camera_index=1.5)
    with pytest.raises(ValueError):
        GestureSettings(pinch_close_ratio=0.6, pinch_release_ratio=0.5)
    with pytest.raises(ValueError):
        GestureSettings(draw_min_alpha=0.8, draw_max_alpha=0.4)
    with pytest.raises(ValueError):
        GestureSettings(depth_alpha=0)
    with pytest.raises(ValueError):
        GestureSettings(depth_sensitivity=4)
    with pytest.raises(ValueError):
        GestureSettings(selection_radius_ratio=0.2)
    with pytest.raises(ValueError):
        GestureSettings(depth_alpha=float("nan"))
    with pytest.raises(ValueError):
        GestureSettings(calibrated=1)

    settings = GestureSettings(selection_radius_ratio=0.025)
    assert settings.selection_radius(1280) == 32
    assert settings.selection_radius(100) == 18
    assert settings.selection_radius(4000) == 56
    with pytest.raises(ValueError):
        settings.selection_radius(0)


def test_default_settings_path_uses_local_app_data(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_settings_path() == tmp_path / "gesture-canvas" / "settings.json"
