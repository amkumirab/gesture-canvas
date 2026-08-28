from gesture_canvas.canvas import DrawingCanvas
from gesture_canvas.interaction import GrabCoordinator, PinchHand


def create_canvas_with_line() -> DrawingCanvas:
    canvas = DrawingCanvas(240, 180)
    canvas.add_point((80, 90), (255, 0, 0), 7)
    canvas.add_point((120, 90), (255, 0, 0), 7)
    canvas.end_stroke()
    return canvas


def test_coordinator_moves_then_transforms_then_returns_to_one_hand():
    canvas = create_canvas_with_line()
    coordinator = GrabCoordinator()
    first = PinchHand("left", (100, 90), pinching=True, started=True)
    assert coordinator.update(canvas, [first])
    assert canvas.is_moving

    first = PinchHand("left", (110, 90), pinching=True)
    coordinator.update(canvas, [first])
    second = PinchHand("right", (150, 90), pinching=True, started=True)
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
    assert canvas.spatial.is_transforming

    second = PinchHand("right", (190, 60), pinching=True)
    coordinator.update(canvas, [first, second])
    scale, _, _, rotation_z = canvas.spatial_transform_info
    assert scale > 1
    assert rotation_z != 0
