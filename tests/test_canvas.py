from pathlib import Path

import numpy as np

from gesture_canvas.canvas import DrawingCanvas


def test_draw_undo_and_redo():
    canvas = DrawingCanvas(100, 80)
    canvas.add_point((10, 10), (255, 0, 0), 5)
    canvas.add_point((50, 20), (255, 0, 0), 5)
    canvas.end_stroke()
    assert np.any(canvas.mask)

    assert canvas.undo()
    assert not np.any(canvas.mask)
    assert canvas.redo()
    assert np.any(canvas.mask)


def test_eraser_removes_pixels():
    canvas = DrawingCanvas(100, 80)
    canvas.add_point((10, 10), (0, 0, 0), 12)
    canvas.add_point((50, 10), (0, 0, 0), 12)
    canvas.end_stroke()
    before = int(canvas.mask.sum())
    canvas.erase_point((30, 10), 18)
    canvas.end_stroke()
    assert int(canvas.mask.sum()) < before


def test_export_uses_white_background(tmp_path: Path):
    canvas = DrawingCanvas(20, 10)
    path = canvas.export(tmp_path / "drawing.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_move_repositions_only_selected_component():
    canvas = DrawingCanvas(120, 80)
    canvas.add_point((15, 20), (255, 0, 0), 5)
    canvas.add_point((35, 20), (255, 0, 0), 5)
    canvas.end_stroke()
    canvas.add_point((80, 50), (0, 255, 0), 5)
    canvas.add_point((100, 50), (0, 255, 0), 5)
    canvas.end_stroke()

    assert canvas.begin_move((20, 20))
    assert canvas.update_move((40, 30))
    assert canvas.end_move()
    assert canvas.mask[20, 20] == 0
    assert canvas.mask[30, 40] > 0
    assert canvas.mask[50, 90] > 0


def test_move_can_grab_near_a_stroke():
    canvas = DrawingCanvas(100, 60)
    canvas.add_point((20, 20), (255, 0, 0), 3)
    canvas.add_point((40, 20), (255, 0, 0), 3)
    canvas.end_stroke()
    assert canvas.begin_move((30, 27), selection_radius=10)


def test_move_is_undoable_and_redoable():
    canvas = DrawingCanvas(100, 60)
    canvas.add_point((10, 20), (255, 0, 0), 5)
    canvas.add_point((30, 20), (255, 0, 0), 5)
    canvas.end_stroke()
    original = canvas.mask.copy()

    assert canvas.begin_move((20, 20))
    canvas.update_move((50, 30))
    canvas.end_move()
    moved = canvas.mask.copy()
    assert not np.array_equal(moved, original)

    assert canvas.undo()
    np.testing.assert_array_equal(canvas.mask, original)
    assert canvas.redo()
    np.testing.assert_array_equal(canvas.mask, moved)


def test_move_rejects_empty_area_and_invalid_radius():
    canvas = DrawingCanvas(100, 60)
    assert not canvas.begin_move((50, 30))
    with np.testing.assert_raises(ValueError):
        canvas.begin_move((50, 30), selection_radius=-1)


def test_large_tracking_jump_starts_a_new_segment():
    canvas = DrawingCanvas(200, 100)
    canvas.add_point((10, 20), (255, 0, 0), 5, max_segment_length=50)
    canvas.add_point((150, 20), (255, 0, 0), 5, max_segment_length=50)
    canvas.end_stroke()
    assert canvas.mask[20, 10] > 0
    assert canvas.mask[20, 80] == 0
    assert canvas.mask[20, 150] > 0
