import pytest

from gesture_canvas.gestures import (
    DrawGestureStabilizer,
    Gesture,
    PinchDetector,
    RuleBasedGestureClassifier,
)
from gesture_canvas.landmarks import Landmark


def make_hand(extended: tuple[bool, bool, bool, bool], pinch: bool = False):
    hand = [Landmark(0.54, 0.8, 0) for _ in range(21)]
    fingers = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    for x, is_extended, (mcp, pip, tip) in zip(
        (0.42, 0.50, 0.58, 0.66), extended, fingers
    ):
        hand[mcp] = Landmark(x, 0.65)
        hand[pip] = Landmark(x, 0.50)
        hand[tip] = Landmark(x, 0.30 if is_extended else 0.68)
    hand[4] = Landmark(0.12 if not pinch else hand[8].x + 0.01, hand[8].y)
    if pinch:
        hand[4] = Landmark(hand[8].x + 0.01, hand[8].y)
    return hand


def test_index_finger_means_draw():
    gesture, confidence = RuleBasedGestureClassifier().classify(make_hand((True, False, False, False)))
    assert gesture is Gesture.DRAW
    assert confidence >= 0.8


def test_two_fingers_mean_erase():
    gesture, _ = RuleBasedGestureClassifier().classify(make_hand((True, True, False, False)))
    assert gesture is Gesture.ERASE


def test_pinch_takes_priority():
    gesture, _ = RuleBasedGestureClassifier().classify(make_hand((True, False, False, False), pinch=True))
    assert gesture is Gesture.PINCH


def test_open_palm_is_not_drawing():
    gesture, _ = RuleBasedGestureClassifier().classify(make_hand((True, True, True, True)))
    assert gesture is Gesture.OPEN_PALM


def test_horizontal_index_finger_still_means_draw():
    hand = make_hand((False, False, False, False))
    hand[0] = Landmark(0.20, 0.50)
    hand[5] = Landmark(0.35, 0.50)
    hand[6] = Landmark(0.50, 0.50)
    hand[8] = Landmark(0.75, 0.50)
    hand[4] = Landmark(0.20, 0.20)
    gesture, _ = RuleBasedGestureClassifier().classify(hand)
    assert gesture is Gesture.DRAW


def test_pinch_hysteresis_prevents_accidental_release():
    detector = PinchDetector(close_ratio=0.34, release_ratio=0.50)
    hand = make_hand((True, False, False, False), pinch=True)
    active, started, ended, _ = detector.update(hand)
    assert (active, started, ended) == (False, False, False)
    active, started, ended, _ = detector.update(hand)
    assert (active, started, ended) == (True, True, False)

    palm_width = abs(hand[17].x - hand[5].x)
    hand[4] = Landmark(hand[8].x + palm_width * 0.42, hand[8].y)
    active, started, ended, _ = detector.update(hand)
    assert (active, started, ended) == (True, False, False)

    hand[4] = Landmark(hand[8].x + palm_width * 0.60, hand[8].y)
    active, started, ended, _ = detector.update(hand)
    assert (active, started, ended) == (True, False, False)
    active, started, ended, _ = detector.update(hand)
    assert (active, started, ended) == (False, False, True)


def test_pinch_threshold_validation():
    with pytest.raises(ValueError):
        PinchDetector(close_ratio=0.5, release_ratio=0.4)
    with pytest.raises(ValueError):
        PinchDetector(close_frames=0)


def test_short_draw_dropout_is_bridged():
    stabilizer = DrawGestureStabilizer(grace_seconds=0.14)
    assert stabilizer.update(Gesture.DRAW, now=1.0) is Gesture.DRAW
    assert stabilizer.update(Gesture.IDLE, now=1.10) is Gesture.DRAW
    assert stabilizer.update(Gesture.IDLE, now=1.20) is Gesture.IDLE


def test_eraser_requires_two_frames_and_then_interrupts_draw():
    stabilizer = DrawGestureStabilizer(grace_seconds=0.14, erase_confirm_frames=2)
    stabilizer.update(Gesture.DRAW, now=1.0)
    assert stabilizer.update(Gesture.ERASE, now=1.01) is Gesture.DRAW
    assert stabilizer.update(Gesture.ERASE, now=1.02) is Gesture.ERASE
    assert not stabilizer.hold_during_missing(now=1.03)


def test_stabilizer_frame_validation():
    with pytest.raises(ValueError):
        DrawGestureStabilizer(erase_confirm_frames=0)
