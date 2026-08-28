import numpy as np

from gesture_canvas.spatial import SpatialScene


def square_contour(
    left: int = 60,
    top: int = 45,
    right: int = 140,
    bottom: int = 115,
) -> np.ndarray:
    return np.asarray(
        [(left, top), (right, top), (right, bottom), (left, bottom)],
        dtype=np.float32,
    )


def test_flat_3d_shape_renders_and_has_selection_bounds():
    scene = SpatialScene(200, 160)
    assert scene.add_plane(square_contour(), (60, 120, 240))

    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    scene.render(frame)

    assert scene.object_count == 1
    assert scene.selected_bounds is not None
    assert np.count_nonzero(frame) > 0


def test_one_hand_move_updates_xyz_from_cursor_and_palm_size():
    scene = SpatialScene(240, 180)
    scene.add_plane(square_contour(), (60, 120, 240))
    original_position = scene.objects[0].position

    assert scene.begin_move((100, 80), depth_signal=0.20)
    assert scene.update_move((125, 95), depth_signal=0.30)
    moved = scene.objects[0].position

    assert moved[:2] == (original_position[0] + 25, original_position[1] + 15)
    assert moved[2] > original_position[2]


def test_depth_dead_zone_ignores_small_palm_jitter():
    scene = SpatialScene(240, 180)
    scene.add_plane(square_contour(), (60, 120, 240))
    assert scene.begin_move((100, 80), depth_signal=0.20)

    assert not scene.update_move((100, 80), depth_signal=0.204)
    assert scene.selected_z == 0


def test_positive_z_appears_larger_and_is_drawn_in_front():
    scene = SpatialScene(240, 180)
    scene.add_plane(square_contour(), (255, 0, 0), z_position=-150)
    far_bounds = scene.selected_bounds
    scene.add_plane(square_contour(), (0, 255, 0), z_position=120)
    near_bounds = scene.selected_bounds

    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    scene.render(frame)

    assert near_bounds[2] > far_bounds[2]
    np.testing.assert_array_equal(frame[80, 100], (0, 255, 0))


def test_two_hand_transform_scales_spins_and_tilts_shape():
    scene = SpatialScene(240, 180)
    scene.add_plane(square_contour(), (60, 120, 240))
    assert scene.begin_move((100, 80))
    assert scene.begin_transform((80, 60), (120, 60))

    assert scene.update_transform((70, 80), (150, 40))
    scale, rotation_x, rotation_y, rotation_z = scene.transform_info

    assert scale > 2
    assert rotation_x == 0
    assert rotation_y != 0
    assert rotation_z != 0


def test_two_hand_depth_difference_creates_visible_y_tilt():
    scene = SpatialScene(240, 180)
    scene.add_plane(square_contour(), (60, 120, 240))
    assert scene.begin_move((100, 80))
    assert scene.begin_transform(
        (70, 70),
        (150, 70),
        first_depth_signal=0.20,
        second_depth_signal=0.20,
    )

    assert scene.update_transform(
        (70, 70),
        (150, 70),
        first_depth_signal=0.28,
        second_depth_signal=0.18,
    )
    scale, rotation_x, rotation_y, rotation_z = scene.transform_info
    assert scale == 1
    assert rotation_x == 0
    assert rotation_y > 30
    assert rotation_z == 0


def test_two_hand_transform_filters_small_jitter():
    scene = SpatialScene(240, 180)
    scene.add_plane(square_contour(), (60, 120, 240))
    assert scene.begin_move((100, 80))
    assert scene.begin_transform((70, 70), (150, 70))

    assert not scene.update_transform((71, 72), (151, 72))
    assert scene.transform_info == (1.0, 0.0, 0.0, 0.0)


def test_z_position_is_clamped_and_snapshot_restores_it():
    scene = SpatialScene(200, 160)
    scene.add_plane(square_contour(), (60, 120, 240), z_position=40)
    snapshot = scene.snapshot()

    assert scene.adjust_z(5_000)
    assert scene.selected_z == scene.focal_length * 0.6
    scene.restore(snapshot)
    assert scene.selected_z == 40


def test_selected_guide_uses_projected_local_axes_and_scene_depth_range():
    scene = SpatialScene(240, 180)
    scene.add_plane(square_contour(), (60, 120, 240), z_position=40)

    face_on = scene.selected_guide()
    assert face_on.origin == (100, 80)
    assert face_on.x_axis[0] > face_on.origin[0]
    assert face_on.y_axis[1] > face_on.origin[1]
    assert face_on.z_axis == face_on.origin
    assert (face_on.minimum_z, face_on.maximum_z) == scene.z_limits

    scene.objects[0].rotation_y = 45
    tilted = scene.selected_guide()
    assert tilted.z_axis != tilted.origin


def test_selected_guide_requires_a_positive_axis_length():
    scene = SpatialScene(200, 160)
    scene.add_plane(square_contour(), (60, 120, 240))
    with np.testing.assert_raises(ValueError):
        scene.selected_guide(axis_length=0)


def test_invalid_plane_is_rejected():
    scene = SpatialScene(200, 160)
    assert not scene.add_plane(np.asarray([(10, 10), (20, 20)]), (0, 0, 0))
