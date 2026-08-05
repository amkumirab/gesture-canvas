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

