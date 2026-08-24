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

    coordinator.update(canvas, [first])
    assert canvas.is_moving
    assert not canvas.is_transforming
    bounds_before = canvas.move_bounds
    first = PinchHand("left", (120, 90), pinching=True)
    coordinator.update(canvas, [first])
    assert canvas.move_bounds != bounds_before

    coordinator.update(canvas, [])
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


def test_coordinator_manipulates_an_extruded_shape():
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
    assert canvas.extrude_at((120, 90))

    coordinator = GrabCoordinator()
    first = PinchHand("left", (120, 90), pinching=True, started=True)
    assert coordinator.update(canvas, [first])
    assert canvas.spatial.is_moving

    second = PinchHand("right", (170, 90), pinching=True, started=True)
    coordinator.update(canvas, [first, second])
    assert canvas.spatial.is_transforming

    second = PinchHand("right", (190, 60), pinching=True)
    coordinator.update(canvas, [first, second])
    scale, _, _, rotation_z = canvas.spatial_transform_info
    assert scale > 1
    assert rotation_z != 0
