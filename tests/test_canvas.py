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


def test_two_hand_transform_scales_selected_component():
    canvas = DrawingCanvas(220, 160)
    canvas.add_point((80, 80), (255, 0, 0), 7)
    canvas.add_point((120, 80), (255, 0, 0), 7)
    canvas.end_stroke()
    assert canvas.begin_move((100, 80))
    assert canvas.begin_transform((80, 60), (120, 60))
    assert canvas.update_transform((60, 60), (140, 60))
    bounds = canvas.move_bounds
    assert bounds is not None
    assert bounds[2] >= 75
    scale, rotation = canvas.transform_info
    assert scale == 2.0
    assert abs(rotation) < 1e-6


def test_two_hand_transform_rotates_horizontal_component():
    canvas = DrawingCanvas(220, 180)
    canvas.add_point((80, 90), (255, 0, 0), 7)
    canvas.add_point((140, 90), (255, 0, 0), 7)
    canvas.end_stroke()
    assert canvas.begin_move((110, 90))
    assert canvas.begin_transform((90, 70), (130, 70))
    assert canvas.update_transform((110, 50), (110, 90))
    bounds = canvas.move_bounds
    assert bounds is not None
    assert bounds[3] > bounds[2]
    _, rotation = canvas.transform_info
    assert rotation == 90.0


def test_transform_is_undoable_and_one_hand_can_continue_without_jump():
    canvas = DrawingCanvas(220, 180)
    canvas.add_point((80, 90), (255, 0, 0), 7)
    canvas.add_point((120, 90), (255, 0, 0), 7)
    canvas.end_stroke()
    original = canvas.mask.copy()
    assert canvas.begin_move((100, 90))
    assert canvas.begin_transform((80, 60), (120, 60))
    assert canvas.update_transform((70, 60), (130, 60))
    before_handoff = canvas.mask.copy()
    assert canvas.end_transform(remaining_anchor=(70, 60))
    assert not canvas.update_move((70, 60))
    np.testing.assert_array_equal(canvas.mask, before_handoff)
    assert canvas.update_move((80, 60))
    canvas.end_move()
    assert canvas.undo()
    np.testing.assert_array_equal(canvas.mask, original)


def test_transform_rejects_too_close_points_and_invalid_scale():
    canvas = DrawingCanvas(100, 80)
    canvas.add_point((20, 20), (255, 0, 0), 5)
    canvas.add_point((40, 20), (255, 0, 0), 5)
    canvas.end_stroke()
    assert canvas.begin_move((30, 20))
    assert not canvas.begin_transform((10, 10), (15, 10), minimum_distance=10)
    with np.testing.assert_raises(ValueError):
        canvas.begin_transform((10, 10), (30, 10), minimum_distance=0)
    assert canvas.begin_transform((10, 10), (40, 10))
    with np.testing.assert_raises(ValueError):
        canvas.update_transform((10, 10), (40, 10), minimum_scale=2, maximum_scale=1)


def test_second_transform_continues_from_existing_scale_and_rotation():
    canvas = DrawingCanvas(260, 220)
    canvas.add_point((100, 110), (255, 0, 0), 7)
    canvas.add_point((140, 110), (255, 0, 0), 7)
    canvas.end_stroke()
    assert canvas.begin_move((120, 110))
    assert canvas.begin_transform((100, 80), (140, 80))
    canvas.update_transform((80, 80), (160, 80))
    canvas.end_transform(remaining_anchor=(80, 80))
    first_scale, first_rotation = canvas.transform_info
    assert first_scale == 2.0

    assert canvas.begin_transform((80, 80), (160, 80))
    canvas.update_transform((120, 40), (120, 120))
    second_scale, second_rotation = canvas.transform_info
    assert second_scale == first_scale
    assert second_rotation == first_rotation + 90.0


def draw_polygon(
    canvas: DrawingCanvas,
    points: list[tuple[int, int]],
    color: tuple[int, int, int] = (40, 80, 220),
    thickness: int = 5,
) -> None:
    for point in points:
        canvas.add_point(point, color, thickness)
    canvas.end_stroke()


def test_closed_shape_is_filled_with_its_stroke_color():
    canvas = DrawingCanvas(140, 120)
    color = (40, 80, 220)
    draw_polygon(
        canvas,
        [(30, 30), (100, 30), (100, 90), (30, 90), (30, 30), (31, 30)],
        color,
    )
    assert canvas.mask[60, 65] == 255
    np.testing.assert_array_equal(canvas.strokes[60, 65], color)


def test_open_shape_keeps_its_interior_empty():
    canvas = DrawingCanvas(140, 120)
    draw_polygon(canvas, [(30, 30), (100, 30), (100, 90), (30, 90)])
    assert canvas.mask[60, 65] == 0


def test_tiny_closed_loop_is_not_auto_filled():
    canvas = DrawingCanvas(80, 80)
    draw_polygon(
        canvas,
        [(30, 30), (38, 30), (38, 38), (30, 38), (30, 30), (31, 30)],
        thickness=3,
    )
    assert canvas.mask[34, 34] == 0


def test_auto_fill_is_part_of_the_same_undo_operation():
    canvas = DrawingCanvas(140, 120)
    draw_polygon(
        canvas,
        [(30, 30), (100, 30), (100, 90), (30, 90), (30, 30), (31, 30)],
    )
    filled = canvas.mask.copy()
    assert canvas.undo()
    assert not np.any(canvas.mask)
    assert canvas.redo()
    np.testing.assert_array_equal(canvas.mask, filled)


def test_closed_eraser_motion_does_not_create_a_fill():
    canvas = DrawingCanvas(140, 120)
    for point in [
        (30, 30),
        (100, 30),
        (100, 90),
        (30, 90),
        (30, 30),
        (31, 30),
    ]:
        canvas.erase_point(point, 5)
    canvas.end_stroke()
    assert not np.any(canvas.mask)
