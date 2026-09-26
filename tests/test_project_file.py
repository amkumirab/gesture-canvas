import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from gesture_canvas.canvas import DrawingCanvas
from gesture_canvas.project_file import (
    PROJECT_FORMAT,
    EditableProject,
    ProjectFormatError,
    load_project,
    save_project,
)


def draw_closed_shape(canvas: DrawingCanvas) -> None:
    for point in [
        (20, 20),
        (80, 20),
        (80, 60),
        (20, 60),
        (20, 20),
        (21, 20),
    ]:
        canvas.add_point(point, (40, 80, 220), 5)
    canvas.end_stroke()


def test_editable_project_round_trip_preserves_canvas_3d_and_history(tmp_path: Path):
    canvas = DrawingCanvas(100, 80)
    draw_closed_shape(canvas)
    painted = canvas.mask.copy()
    assert canvas.promote_to_3d((50, 40))
    original_z = canvas.selected_spatial_z
    shape = canvas.spatial.objects[0]
    shape.rotation_x = 18.0
    shape.rotation_y = -12.0
    shape.rotation_z = 31.0
    assert canvas.adjust_spatial_z(20)
    moved_z = canvas.selected_spatial_z
    assert canvas.undo()

    saved = save_project(
        tmp_path / "drawing",
        canvas.project_state(),
        brush_size=11,
        active_tool="red",
    )

    assert saved.suffix == ".gcanvas"
    project = load_project(saved)
    assert isinstance(project, EditableProject)
    assert project.brush_size == 11
    assert project.active_tool == "red"

    restored = DrawingCanvas(100, 80)
    restored.restore_project_state(project.canvas)
    assert restored.spatial_object_count == 1
    assert restored.selected_spatial_z == original_z
    assert restored.spatial_transform_info == (1.0, 18.0, -12.0, 31.0)
    assert restored.can_undo
    assert restored.can_redo

    assert restored.undo()
    assert restored.spatial_object_count == 0
    np.testing.assert_array_equal(restored.mask, painted)
    assert restored.redo()
    assert restored.spatial_object_count == 1
    assert restored.redo()
    assert restored.selected_spatial_z == moved_z


def test_load_project_fits_content_and_history_to_current_canvas(tmp_path: Path):
    canvas = DrawingCanvas(100, 80)
    canvas.add_point((10, 10), (255, 90, 30), 5)
    canvas.add_point((30, 10), (255, 90, 30), 5)
    canvas.end_stroke()
    contour = np.asarray([(40, 25), (70, 25), (55, 55)], dtype=np.float32)
    assert canvas.spatial.add_plane(contour, (50, 60, 240), z_position=12)
    original_position = canvas.spatial.objects[0].position

    path = save_project(
        tmp_path / "resize.gcanvas",
        canvas.project_state(),
        brush_size=7,
        active_tool="blue",
    )
    project = load_project(path, target_size=(200, 160))
    restored = DrawingCanvas(200, 160)
    restored.restore_project_state(project.canvas)

    assert (project.canvas.width, project.canvas.height) == (200, 160)
    assert restored.mask[20, 20] > 0
    resized_position = restored.spatial.objects[0].position
    assert resized_position == pytest.approx(tuple(value * 2 for value in original_position))


def test_unsupported_project_version_is_rejected(tmp_path: Path):
    path = tmp_path / "future.gcanvas"
    manifest = {
        "format": PROJECT_FORMAT,
        "version": 999,
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(ProjectFormatError, match="version"):
        load_project(path)


@pytest.mark.parametrize("payload", [b"", b"not a project", b"{broken"])
def test_invalid_project_is_rejected_with_a_clear_error(tmp_path: Path, payload: bytes):
    path = tmp_path / "broken.gcanvas"
    path.write_bytes(payload)

    with pytest.raises(ProjectFormatError):
        load_project(path)


def test_restore_rejects_a_state_for_different_dimensions():
    source = DrawingCanvas(100, 80)
    target = DrawingCanvas(120, 90)

    with pytest.raises(ValueError, match="dimensions"):
        target.restore_project_state(source.project_state())


def test_save_rejects_an_unknown_tool_without_creating_a_file(tmp_path: Path):
    path = tmp_path / "invalid.gcanvas"

    with pytest.raises(ValueError, match="tool"):
        save_project(path, DrawingCanvas(20, 20).project_state(), 7, "purple")

    assert not path.exists()


def test_project_round_trip_preserves_layer_visibility_and_lock(tmp_path: Path):
    canvas = DrawingCanvas(100, 80)
    contour = np.asarray(
        [(20, 20), (80, 20), (80, 60), (20, 60)],
        dtype=np.float32,
    )
    assert canvas.spatial.add_plane(contour, (40, 80, 220))
    assert canvas.toggle_spatial_visibility()
    assert canvas.toggle_spatial_lock()

    path = save_project(
        tmp_path / "layers.gcanvas",
        canvas.project_state(),
        brush_size=7,
        active_tool="blue",
    )
    project = load_project(path)
    layer = project.canvas.current.spatial.objects[0]

    assert not layer.visible
    assert layer.locked


def test_projects_saved_before_layer_flags_still_open(tmp_path: Path):
    canvas = DrawingCanvas(100, 80)
    contour = np.asarray(
        [(20, 20), (80, 20), (80, 60), (20, 60)],
        dtype=np.float32,
    )
    canvas.spatial.add_plane(contour, (40, 80, 220))
    path = save_project(
        tmp_path / "older.gcanvas",
        canvas.project_state(),
        brush_size=7,
        active_tool="blue",
    )

    with ZipFile(path, "r") as archive:
        contents = {
            name: archive.read(name)
            for name in archive.namelist()
        }
    manifest = json.loads(contents["manifest.json"])
    for snapshot in manifest["snapshots"]:
        for shape in snapshot["objects"]:
            shape.pop("visible")
            shape.pop("locked")
    contents["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)

    project = load_project(path)
    layer = project.canvas.current.spatial.objects[0]
    assert layer.visible
    assert not layer.locked
