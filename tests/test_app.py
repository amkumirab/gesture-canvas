import sys
from pathlib import Path

import pytest

from gesture_canvas.app import open_selected_camera, parse_args


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
