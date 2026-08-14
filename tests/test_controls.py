from gesture_canvas.controls import (
    DEFAULT_BRUSH_SIZE,
    MAX_BRUSH_SIZE,
    MIN_BRUSH_SIZE,
    adjust_brush_size,
)


def test_brush_size_can_be_increased_and_decreased():
    assert adjust_brush_size(DEFAULT_BRUSH_SIZE, 1) == DEFAULT_BRUSH_SIZE + 1
    assert adjust_brush_size(DEFAULT_BRUSH_SIZE, -1) == DEFAULT_BRUSH_SIZE - 1


def test_brush_size_is_clamped_to_supported_range():
    assert adjust_brush_size(MAX_BRUSH_SIZE, 1) == MAX_BRUSH_SIZE
    assert adjust_brush_size(MIN_BRUSH_SIZE, -1) == MIN_BRUSH_SIZE
    assert adjust_brush_size(DEFAULT_BRUSH_SIZE, 100) == MAX_BRUSH_SIZE
    assert adjust_brush_size(DEFAULT_BRUSH_SIZE, -100) == MIN_BRUSH_SIZE
