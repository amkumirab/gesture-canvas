import numpy as np
import pytest

from gesture_canvas.canvas import DrawingCanvas
from gesture_canvas.interaction import GrabCoordinator, PinchHand


def create_canvas_with_line() -> DrawingCanvas:
    canvas = DrawingCanvas(240, 180)
    canvas.add_point((80, 90), (255, 0, 0), 7)
    canvas.add_point((120, 90), (255, 0, 0), 7)
    canvas.end_stroke()
    return canvas


@pytest.mark.parametrize("spatial_mode", ["auto", "tilt_x"])
def test_coordinator_moves_then_transforms_then_returns_to_one_hand(spatial_mode):
    canvas = create_canvas_with_line()
    canvas.spatial.set_transform_mode(spatial_mode)
    coordinator = GrabCoordinator()
    first = PinchHand("left", (100, 90), pinching=True, started=True)
    assert coordinator.update(canvas, [first])
    assert canvas.is_moving

    first = PinchHand("left", (110, 90), pinching=True)
    coordinator.update(canvas, [first])
    second = PinchHand("right", (150, 90), pinching=True, started=True)
    coordinator.update(canvas, [first, second])
    assert coordinator.arming_transform
    coordinator.update(canvas, [first, second])
    assert canvas.is_transforming

    second = PinchHand("right", (190, 90), pinching=True)
    coordinator.update(canvas, [first, second])
    scale, _ = canvas.transform_info
    assert scale > 1.5

    released_second = PinchHand("right", (190, 90), pinching=False)
    coordinator.update(canvas, [first, released_second])
    assert canvas.is_moving
    assert not canvas.is_transforming
    bounds_before = canvas.move_bounds
    first = PinchHand("left", (120, 90), pinching=True)
    coordinator.update(canvas, [first])
    assert canvas.move_bounds != bounds_before

    released_first = PinchHand("left", (120, 90), pinching=False)
    coordinator.update(canvas, [released_first])
    assert not canvas.is_moving


def test_coordinator_ignores_pinch_above_canvas():
    canvas = create_canvas_with_line()
    coordinator = GrabCoordinator()
    hand = PinchHand("left", (100, 20), pinching=True, started=True)
    assert not coordinator.update(canvas, [hand], canvas_top=74)


def test_coordinator_validation_and_reset():
    canvas = create_canvas_with_line()
    coordinator = GrabCoordinator(selection_radius=10)
    coordinator.reset(canvas)
    assert coordinator.hand_ids == []

    try:
        GrabCoordinator(selection_radius=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative selection radius should fail")

    try:
        GrabCoordinator(missing_grace_frames=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative dropout grace should fail")

    try:
        GrabCoordinator(transform_arm_frames=0)
    except ValueError:
        pass
    else:
        raise AssertionError("transform arming must use at least one frame")


def test_coordinator_survives_brief_hand_tracking_dropout():
    canvas = create_canvas_with_line()
    coordinator = GrabCoordinator(missing_grace_frames=2)
    hand = PinchHand("left", (100, 90), pinching=True, started=True)
    assert coordinator.update(canvas, [hand])

    assert coordinator.update(canvas, [])
    assert coordinator.recovering_tracking
    assert canvas.is_moving

    recovered = PinchHand("left", (110, 90), pinching=True)
    assert coordinator.update(canvas, [recovered])
    assert not coordinator.recovering_tracking
    assert canvas.is_moving


def test_coordinator_ends_move_after_dropout_grace_expires():
    canvas = create_canvas_with_line()
    coordinator = GrabCoordinator(missing_grace_frames=1)
    hand = PinchHand("left", (100, 90), pinching=True, started=True)
    assert coordinator.update(canvas, [hand])

    coordinator.update(canvas, [])
    coordinator.update(canvas, [])
    assert not canvas.is_moving


def test_transform_does_not_arm_from_a_cached_primary_hand():
    canvas = create_canvas_with_line()
    coordinator = GrabCoordinator(missing_grace_frames=2)
    first = PinchHand("left", (100, 90), pinching=True, started=True)
    second = PinchHand("right", (150, 90), pinching=True, started=True)
    assert coordinator.update(canvas, [first])
    assert coordinator.update(canvas, [first, second])
    assert coordinator.arming_transform

    assert coordinator.update(canvas, [second])
    assert coordinator.recovering_tracking
    assert not coordinator.arming_transform
    assert not canvas.is_transforming

    recovered = PinchHand("left", (102, 90), pinching=True)
    assert coordinator.update(canvas, [recovered, second])
    assert coordinator.arming_transform
    assert coordinator.update(canvas, [recovered, second])
    assert canvas.is_transforming


def test_coordinator_manipulates_a_spatial_shape_in_xyz():
    canvas = DrawingCanvas(240, 180)
    for point in [
        (60, 45),
        (180, 45),
        (180, 135),
        (60, 135),
        (60, 45),
        (61, 45),
    ]:
        canvas.add_point(point, (60, 120, 240), 7)
    canvas.end_stroke()
    assert canvas.promote_to_3d((120, 90))

    coordinator = GrabCoordinator()
    first = PinchHand(
        "left",
        (120, 90),
        pinching=True,
        started=True,
        depth_signal=0.2,
    )
    assert coordinator.update(canvas, [first])
    assert canvas.spatial.is_moving

    first = PinchHand("left", (125, 95), pinching=True, depth_signal=0.3)
    coordinator.update(canvas, [first])
    assert canvas.selected_spatial_z > 0

    second = PinchHand("right", (170, 90), pinching=True, started=True)
    coordinator.update(canvas, [first, second])
    assert coordinator.arming_transform
    coordinator.update(canvas, [first, second])
    assert canvas.spatial.is_transforming

    second = PinchHand("right", (190, 60), pinching=True)
    coordinator.update(canvas, [first, second])
    scale, _, _, rotation_z = canvas.spatial_transform_info
    assert scale > 1
    assert canvas.spatial_transform_control == "scale"
    assert rotation_z == 0


@pytest.mark.parametrize("mode", ["auto", "spin"])
def test_spatial_transform_freezes_and_rebases_during_hand_dropout(mode):
    canvas = DrawingCanvas(240, 180)
    for point in [
        (60, 45),
        (180, 45),
        (180, 135),
        (60, 135),
        (60, 45),
        (61, 45),
    ]:
        canvas.add_point(point, (60, 120, 240), 7)
    canvas.end_stroke()
    assert canvas.promote_to_3d((120, 90))

    canvas.spatial.set_transform_mode(mode)
    coordinator = GrabCoordinator(missing_grace_frames=3)
    first = PinchHand("left", (100, 90), pinching=True, started=True)
    second = PinchHand("right", (180, 90), pinching=True, started=True)
    assert coordinator.update(canvas, [first])
    assert coordinator.update(canvas, [first, second])
    assert coordinator.arming_transform
    assert coordinator.update(canvas, [first, second])
    turned_second = PinchHand("right", (140, 130), pinching=True)
    coordinator.update(canvas, [first, turned_second])
    rotation_before = canvas.spatial_transform_info[3]
    assert rotation_before != 0

    moved_first = PinchHand("left", (40, 40), pinching=True)
    coordinator.update(canvas, [moved_first])
    assert coordinator.recovering_tracking
    assert canvas.spatial_transform_info[3] == rotation_before

    recovered_second = PinchHand("right", (120, 40), pinching=True)
    coordinator.update(canvas, [moved_first, recovered_second])
    assert not coordinator.recovering_tracking
    assert canvas.spatial_transform_info[3] == rotation_before


def test_manual_control_switch_keeps_two_hand_grab_and_one_undo_operation():
    canvas = DrawingCanvas(240, 180)
    contour = np.asarray([(60, 45), (180, 45), (180, 135), (60, 135)], dtype=np.float32)
    canvas.spatial.add_plane(contour, (60, 120, 240))
    canvas.spatial.set_transform_mode("spin")
    coordinator = GrabCoordinator(transform_arm_frames=1)
    first = PinchHand("left", (100, 90), pinching=True, started=True)
    second = PinchHand("right", (180, 90), pinching=True, started=True)
    coordinator.update(canvas, [first])
    coordinator.update(canvas, [first, second])
    turned = PinchHand("right", (140, 130), pinching=True)
    coordinator.update(canvas, [first, turned])
    before_switch = canvas.spatial_transform_info
    assert before_switch[3] > 0

    canvas.spatial.set_transform_mode("tilt_y")
    coordinator.update(canvas, [first, turned])
    assert canvas.is_transforming
    assert canvas.spatial_transform_info == before_switch
    moved_first = PinchHand("left", (120, 90), pinching=True)
    moved_second = PinchHand("right", (160, 130), pinching=True)
    coordinator.update(canvas, [moved_first, moved_second])
    assert canvas.spatial_transform_info[0:2] == before_switch[0:2]
    assert canvas.spatial_transform_info[2] > before_switch[2]
    assert canvas.spatial_transform_info[3] == before_switch[3]

    released = PinchHand("right", (160, 130), pinching=False)
    coordinator.update(canvas, [moved_first, released])
    assert canvas.is_moving
    assert not canvas.is_transforming
    assert canvas.undo()
    assert canvas.spatial_transform_info == (1.0, 0.0, 0.0, 0.0)
    assert canvas.spatial.transform_mode == "tilt_y"


def test_releasing_second_hand_immediately_after_mode_switch_does_not_move_layer():
    canvas = DrawingCanvas(240, 180)
    contour = np.asarray([(60, 45), (180, 45), (180, 135), (60, 135)], dtype=np.float32)
    canvas.spatial.add_plane(contour, (60, 120, 240))
    canvas.spatial.set_transform_mode("spin")
    coordinator = GrabCoordinator(transform_arm_frames=1)
    first = PinchHand("left", (100, 90), pinching=True, started=True)
    second = PinchHand("right", (180, 90), pinching=True, started=True)
    coordinator.update(canvas, [first])
    coordinator.update(canvas, [first, second])
    moved_first = PinchHand("left", (80, 70), pinching=True)
    turned = PinchHand("right", (140, 130), pinching=True)
    coordinator.update(canvas, [moved_first, turned])
    before_position = canvas.spatial.objects[0].position
    before_rotation = canvas.spatial_transform_info

    canvas.spatial.set_transform_mode("tilt_y")
    released = PinchHand("right", (140, 130), pinching=False)
    coordinator.update(canvas, [moved_first, released])

    assert not canvas.is_transforming
    assert canvas.spatial.objects[0].position == before_position
    assert canvas.spatial_transform_info == before_rotation
