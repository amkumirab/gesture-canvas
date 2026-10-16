import sys
from pathlib import Path

import numpy as np
import pytest

from gesture_canvas.app import (
    apply_layer_action,
    camera_recording_fps,
    frame_for_recording,
    is_record_toggle_key,
    open_selected_camera,
    parse_args,
    startup_project_path,
)
from gesture_canvas.canvas import DrawingCanvas
from gesture_canvas.layer_panel import LayerPanelTarget


def test_explicit_camera_is_opened_without_fallback(monkeypatch):
    opened = []

    def fake_open(index):
        opened.append(index)
        return f"camera-{index}"

    monkeypatch.setattr("gesture_canvas.app.open_camera", fake_open)
    assert open_selected_camera(2, 0) == ("camera-2", 2)
    assert opened == [2]


def test_stale_saved_camera_falls_back_to_zero(monkeypatch):
    opened = []

    def fake_open(index):
        opened.append(index)
        if index == 3:
            raise RuntimeError("missing")
        return f"camera-{index}"

    monkeypatch.setattr("gesture_canvas.app.open_camera", fake_open)
    assert open_selected_camera(None, 3) == ("camera-0", 0)
    assert opened == [3, 0]


def test_explicit_camera_failure_is_not_hidden(monkeypatch):
    def fail(_index):
        raise RuntimeError("missing")

    monkeypatch.setattr("gesture_canvas.app.open_camera", fail)
    with pytest.raises(RuntimeError):
        open_selected_camera(3, 0)
    with pytest.raises(ValueError):
        open_selected_camera(-1, 0)


def test_project_can_be_selected_at_startup(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gesture-canvas", "--project", "drawing.gcanvas"],
    )

    assert parse_args().project == Path("drawing.gcanvas")


def test_previous_session_restore_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gesture-canvas", "--no-restore"],
    )

    assert parse_args().no_restore


def test_explicit_project_takes_priority_over_session_recovery(tmp_path):
    recovery = tmp_path / "recovery.gcanvas"
    recovery.touch()
    requested = tmp_path / "drawing.gcanvas"

    assert startup_project_path(requested, recovery, True) == requested


def test_session_recovery_requires_an_existing_file_and_restore_flag(tmp_path):
    recovery = tmp_path / "recovery.gcanvas"
    assert startup_project_path(None, recovery, True) is None
    recovery.touch()
    assert startup_project_path(None, recovery, True) == recovery
    assert startup_project_path(None, recovery, False) is None


def test_layer_panel_actions_return_clear_feedback():
    canvas = DrawingCanvas(100, 80)
    contour = np.asarray(
        [(20, 20), (80, 20), (80, 60), (20, 60)],
        dtype=np.float32,
    )
    canvas.spatial.add_plane(contour, (40, 80, 220))

    assert apply_layer_action(LayerPanelTarget("lock", "LOCK"), canvas) == "Layer locked"
    assert apply_layer_action(LayerPanelTarget("duplicate", "COPY"), canvas) == "Layer copied"
    assert canvas.spatial_object_count == 2


def test_record_mode_can_be_selected_at_startup(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gesture-canvas", "--record-mode", "canvas"],
    )
    assert parse_args().record_mode == "canvas"


@pytest.mark.parametrize("key", [ord("v"), ord("V"), 0x780000, 65478, 63244])
def test_record_shortcut_supports_common_backends(key):
    assert is_record_toggle_key(key)
    assert not is_record_toggle_key(ord("r"))


def test_camera_recording_fps_uses_valid_value_or_fallback():
    class Camera:
        def __init__(self, fps):
            self.fps = fps

        def get(self, _property):
            return self.fps

    assert camera_recording_fps(Camera(60)) == 60
    assert camera_recording_fps(Camera(0)) == 30
    assert camera_recording_fps(object()) == 30


def test_clean_recording_frame_excludes_camera_view():
    canvas = DrawingCanvas(40, 30)
    canvas.add_point((5, 10), (20, 30, 40), 3)
    canvas.add_point((20, 10), (20, 30, 40), 3)
    canvas.end_stroke()
    display = np.full((30, 40, 3), 90, dtype=np.uint8)

    assert frame_for_recording(canvas, display, "camera") is display
    clean = frame_for_recording(canvas, display, "canvas")
    assert np.all(clean[0, 0] == 255)
    assert not np.all(clean[10, 10] == 255)
    with pytest.raises(ValueError, match="mode"):
        frame_for_recording(canvas, display, "unknown")
