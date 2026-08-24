import numpy as np

from gesture_canvas.spatial import SpatialScene


def square_contour() -> np.ndarray:
    return np.asarray(
        [(60, 45), (140, 45), (140, 115), (60, 115)],
        dtype=np.float32,
    )


def test_extruded_shape_renders_and_has_selection_bounds():
    scene = SpatialScene(200, 160)
    assert scene.add_extrusion(square_contour(), (60, 120, 240))

    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    scene.render(frame)

    assert scene.object_count == 1
    assert scene.selected_bounds is not None
    assert np.count_nonzero(frame) > 0


def test_one_hand_move_repositions_extruded_shape():
    scene = SpatialScene(240, 180)
    scene.add_extrusion(square_contour(), (60, 120, 240))
    original_position = scene.objects[0].position

    assert scene.begin_move((100, 80))
    assert scene.update_move((125, 95))
    assert scene.objects[0].position == (
        original_position[0] + 25,
        original_position[1] + 15,
    )


def test_two_hand_transform_scales_spins_and_tilts_shape():
    scene = SpatialScene(240, 180)
    scene.add_extrusion(square_contour(), (60, 120, 240))
    assert scene.begin_move((100, 80))
    assert scene.begin_transform((80, 60), (120, 60))

    assert scene.update_transform((70, 80), (150, 40))
    scale, rotation_x, rotation_y, rotation_z = scene.transform_info

    assert scale > 2
    assert rotation_y != 24
    assert rotation_z != 0
    assert rotation_x == -18


def test_depth_is_clamped_and_snapshot_restores_it():
    scene = SpatialScene(200, 160)
    scene.add_extrusion(square_contour(), (60, 120, 240), depth=40)
    snapshot = scene.snapshot()

    assert scene.adjust_depth(500)
    assert scene.selected_depth == 240
    scene.restore(snapshot)
    assert scene.selected_depth == 40


def test_invalid_extrusion_is_rejected():
    scene = SpatialScene(200, 160)
    assert not scene.add_extrusion(np.asarray([(10, 10), (20, 20)]), (0, 0, 0))
