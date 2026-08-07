from gesture_canvas.app import drag_gesture_for_tool
from gesture_canvas.gestures import Gesture


def test_move_tool_uses_pointing_gesture():
    assert drag_gesture_for_tool("move") is Gesture.DRAW


def test_grab_tool_uses_pinch_gesture():
    assert drag_gesture_for_tool("grab") is Gesture.PINCH


def test_drawing_tools_do_not_start_dragging():
    assert drag_gesture_for_tool("blue") is None
    assert drag_gesture_for_tool("eraser") is None
