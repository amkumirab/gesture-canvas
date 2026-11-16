"""Exercise the real application loop with scripted hand input and no webcam."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gesture_canvas import app
from gesture_canvas.canvas import DrawingCanvas
from gesture_canvas.gestures import Gesture
from gesture_canvas.transform_menu import TransformControlMenu


@pytest.mark.parametrize(
    ("menu_key", "expected_mode"),
    [(-1, "spin"), (27, "auto"), (ord("3"), "tilt_y")],
)
def test_menu_blocks_paint_erase_and_grabs_until_a_neutral_hand(
    monkeypatch, tmp_path, menu_key, expected_mode,
):
    menu = TransformControlMenu(640, 480)

    def center(rect):
        left, top, right, bottom = rect
        return (left + right) // 2, (top + bottom) // 2

    trigger = center(menu.trigger_rect)
    spin = center(menu.option_rects["spin"])
    events = [
        (0.0, trigger, Gesture.DRAW),
        (0.7, trigger, Gesture.DRAW),
        (1.0, (200, 200), Gesture.PINCH),
        (1.1, spin, Gesture.DRAW),
        (1.8, spin, Gesture.DRAW),
        (2.0, (300, 200), Gesture.DRAW),
        (2.1, (300, 200), Gesture.ERASE),
        (2.2, (200, 200), Gesture.PINCH),
        (2.3, (300, 200), Gesture.IDLE),
        (2.4, (300, 200), Gesture.DRAW),
    ]
    state = {"index": -1}
    canvases = []

    def current():
        return events[max(0, state["index"])]

    class Camera:
        released = False

        def read(self):
            state["index"] += 1
            assert state["index"] < len(events), "Application did not exit on Q"
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def get(self, _property):
            return 30.0

        def release(self):
            self.released = True

    class ObservedCanvas(DrawingCanvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.draw_calls = self.erase_calls = self.grab_calls = 0
            contour = np.asarray([(80, 150), (380, 150), (380, 350), (80, 350)], dtype=np.float32)
            self.spatial.add_plane(contour, (50, 60, 240))
            canvases.append(self)

        def add_point(self, *args, **kwargs):
            self.draw_calls += 1
            return super().add_point(*args, **kwargs)

        def erase_point(self, *args, **kwargs):
            self.erase_calls += 1
            return super().erase_point(*args, **kwargs)

        def begin_move(self, *args, **kwargs):
            self.grab_calls += 1
            return super().begin_move(*args, **kwargs)

    camera = Camera()
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    hand = SimpleNamespace(landmarks=landmarks, handedness="left", confidence=1.0)
    monkeypatch.setattr(sys, "argv", [
        "gesture-canvas", "--settings", str(tmp_path / "settings.json"),
        "--output", str(tmp_path / "output"), "--no-restore",
    ])
    monkeypatch.setattr(app, "open_camera", lambda _index: camera)
    monkeypatch.setattr(app, "create_hand_tracker", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(app, "detect_hands", lambda *_args: [hand])
    monkeypatch.setattr(app, "DrawingCanvas", ObservedCanvas)
    monkeypatch.setattr(app, "GestureRecognizer", lambda *_args: SimpleNamespace(
        recognize=lambda _points: (current()[2], 1.0),
    ))
    monkeypatch.setattr(app, "PinchDetector", lambda **_kwargs: SimpleNamespace(
        update=lambda _points: (current()[2] is Gesture.PINCH, current()[2] is Gesture.PINCH, False, 1.0),
    ))
    monkeypatch.setattr(app, "DrawGestureStabilizer", lambda: SimpleNamespace(
        update=lambda gesture, _now: gesture,
    ))
    smoother = lambda **_kwargs: SimpleNamespace(update=lambda value: value, reset=lambda: None)
    monkeypatch.setattr(app, "AdaptiveSmoother", smoother)
    monkeypatch.setattr(app, "ScalarSmoother", smoother)
    monkeypatch.setattr(app, "to_pixel", lambda *_args: current()[1])
    monkeypatch.setattr(app, "palm_span", lambda _points: 0.2)
    monkeypatch.setattr(app.time, "monotonic", lambda: current()[0])
    monkeypatch.setattr(app.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)
    def next_key(_delay):
        if state["index"] == len(events) - 1:
            return ord("q")
        return menu_key if state["index"] == 3 else -1

    monkeypatch.setattr(app.cv2, "waitKeyEx", next_key)

    app.main()

    canvas = canvases[0]
    assert canvas.spatial.transform_mode == expected_mode
    assert canvas.draw_calls == 1
    assert canvas.erase_calls == 0
    assert canvas.grab_calls == 0
    assert camera.released
