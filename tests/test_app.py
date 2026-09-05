import sys
from pathlib import Path

import pytest

from gesture_canvas.app import open_selected_camera, parse_args, startup_project_path


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
