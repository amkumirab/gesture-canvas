from gesture_canvas.controls import (
    DEFAULT_BRUSH_SIZE,
    MAX_BRUSH_SIZE,
    MIN_BRUSH_SIZE,
    adjust_brush_size,
    transform_mode_for_key,
)


def test_brush_size_can_be_increased_and_decreased():
    assert adjust_brush_size(DEFAULT_BRUSH_SIZE, 1) == DEFAULT_BRUSH_SIZE + 1
    assert adjust_brush_size(DEFAULT_BRUSH_SIZE, -1) == DEFAULT_BRUSH_SIZE - 1


def test_brush_size_is_clamped_to_supported_range():
    assert adjust_brush_size(MAX_BRUSH_SIZE, 1) == MAX_BRUSH_SIZE
    assert adjust_brush_size(MIN_BRUSH_SIZE, -1) == MIN_BRUSH_SIZE
    assert adjust_brush_size(DEFAULT_BRUSH_SIZE, 100) == MAX_BRUSH_SIZE
    assert adjust_brush_size(DEFAULT_BRUSH_SIZE, -100) == MIN_BRUSH_SIZE


def test_number_keys_select_spatial_transform_modes_only():
    expected = ["auto", "scale", "tilt_x", "tilt_y", "spin"]
    for key, mode in zip("01234", expected):
        assert transform_mode_for_key(ord(key)) == mode
    for key in (-1, ord("z"), ord("y"), ord("v"), ord("5")):
        assert transform_mode_for_key(key) is None
