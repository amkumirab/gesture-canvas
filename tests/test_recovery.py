from pathlib import Path

import pytest

from gesture_canvas.canvas import DrawingCanvas
from gesture_canvas.project_file import load_project
from gesture_canvas.recovery import AutosaveController


def test_autosave_runs_only_when_due_and_content_changed(tmp_path: Path):
    path = tmp_path / "recovery.gcanvas"
    canvas = DrawingCanvas(80, 60)
    canvas.add_point((10, 10), (255, 90, 30), 5)
    controller = AutosaveController(path, interval_seconds=5)
    try:
        assert not controller.maybe_save(canvas, 7, "blue", now=0)
        assert not controller.maybe_save(canvas, 7, "blue", now=4.9)
        assert controller.maybe_save(canvas, 7, "blue", now=5)
        controller.flush()

        assert path.exists()
        assert controller.saved_revision == canvas.revision
        assert not controller.maybe_save(canvas, 7, "blue", now=10)
    finally:
        controller.close()


def test_autosave_snapshot_does_not_end_an_active_stroke(tmp_path: Path):
    canvas = DrawingCanvas(80, 60)
    canvas.add_point((10, 20), (255, 90, 30), 5)
    controller = AutosaveController(tmp_path / "recovery.gcanvas", interval_seconds=1)
    try:
        controller.maybe_save(canvas, 7, "blue", now=0)
        assert controller.maybe_save(canvas, 7, "blue", now=1)
        controller.flush()

        canvas.add_point((30, 20), (255, 90, 30), 5)
        assert canvas.mask[20, 20] > 0
    finally:
        controller.close()


def test_workspace_only_change_is_saved_and_restored(tmp_path: Path):
    path = tmp_path / "recovery.gcanvas"
    canvas = DrawingCanvas(80, 60)
    controller = AutosaveController(path, interval_seconds=1)
    try:
        controller.maybe_save(canvas, 7, "blue", now=0)
        assert controller.maybe_save(canvas, 7, "blue", now=1)
        controller.flush()
        assert controller.maybe_save(canvas, 12, "red", now=2)
        controller.flush()
    finally:
        controller.close()

    recovered = load_project(path)
    assert recovered.brush_size == 12
    assert recovered.active_tool == "red"


def test_background_failure_is_reported_once_without_raising(tmp_path: Path):
    def fail_save(*_args):
        raise OSError("disk unavailable")

    canvas = DrawingCanvas(40, 30)
    controller = AutosaveController(
        tmp_path / "recovery.gcanvas",
        interval_seconds=1,
        save_function=fail_save,
    )
    try:
        controller.maybe_save(canvas, 7, "blue", now=0)
        assert controller.maybe_save(canvas, 7, "blue", now=1)
        controller.flush()

        assert controller.consume_error() == "disk unavailable"
        assert controller.consume_error() is None
    finally:
        controller.close()


def test_save_now_captures_latest_state_and_close_is_idempotent(tmp_path: Path):
    path = tmp_path / "recovery.gcanvas"
    canvas = DrawingCanvas(80, 60)
    canvas.add_point((10, 10), (60, 210, 70), 5)
    canvas.add_point((35, 10), (60, 210, 70), 5)
    controller = AutosaveController(path)

    assert controller.save_now(canvas, 9, "green") == path
    controller.close()
    controller.close()

    recovered = load_project(path)
    assert recovered.canvas.current.mask[10, 20] > 0
    assert recovered.brush_size == 9
    assert recovered.canvas.undo == []
    assert recovered.canvas.redo == []
    with pytest.raises(RuntimeError, match="closed"):
        controller.maybe_save(canvas, 9, "green", now=20)


def test_autosave_interval_must_be_positive(tmp_path: Path):
    with pytest.raises(ValueError, match="positive"):
        AutosaveController(tmp_path / "recovery.gcanvas", interval_seconds=0)
