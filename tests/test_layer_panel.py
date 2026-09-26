import numpy as np
import pytest

from gesture_canvas.layer_panel import LayerPanel
from gesture_canvas.spatial import SpatialScene


def layer_data():
    scene = SpatialScene(640, 480)
    contour = np.asarray(
        [(80, 90), (150, 90), (150, 150), (80, 150)],
        dtype=np.float32,
    )
    scene.add_plane(contour, (50, 60, 240))
    scene.add_plane(contour + (80, 40), (60, 210, 70))
    return scene.layer_info(), scene.selected_index


def midpoint(rect):
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2


def test_layer_rows_and_actions_are_hit_testable():
    panel = LayerPanel(640, 480)
    layers, selected = layer_data()

    first_index, first_rect = panel._layer_rects(len(layers), selected)[0]
    row = panel.hit_test(midpoint(first_rect), layers, selected)
    assert row.key == f"select:{first_index}"

    action_rects = panel._action_rects(layers[selected])
    for key, _, rect in action_rects:
        assert panel.hit_test(midpoint(rect), layers, selected).key == key


def test_panel_draws_layer_metadata_and_hover_progress():
    panel = LayerPanel(640, 480)
    layers, selected = layer_data()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    panel.draw(frame, layers, selected, hover_key="duplicate", hover_progress=0.5)

    assert np.count_nonzero(frame) > 0
    assert panel.contains((panel.left + 2, panel.top + 2))
    assert not panel.contains((10, 10))


def test_panel_handles_empty_layers_and_validates_dimensions():
    panel = LayerPanel(320, 240)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    panel.draw(frame, (), None)
    assert panel.hit_test((panel.left + 10, panel.top + 60), (), None) is None
    with pytest.raises(ValueError, match="dimensions"):
        LayerPanel(0, 240)
    with pytest.raises(ValueError, match="dimensions"):
        panel.draw(np.zeros((200, 320, 3), dtype=np.uint8), (), None)
