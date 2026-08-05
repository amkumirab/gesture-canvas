from gesture_canvas.gestures import Gesture, RuleBasedGestureClassifier
from gesture_canvas.landmarks import Landmark


def make_hand(extended: tuple[bool, bool, bool, bool], pinch: bool = False):
    hand = [Landmark(0.5, 0.7, 0) for _ in range(21)]
    hand[5] = Landmark(0.4, 0.6)
    hand[17] = Landmark(0.7, 0.6)
    hand[4] = Landmark(0.2 if not pinch else 0.51, 0.5)
    hand[8] = Landmark(0.5, 0.5)
    for is_extended, (tip, pip) in zip(extended, ((8, 6), (12, 10), (16, 14), (20, 18))):
        hand[pip] = Landmark(0.5, 0.55)
        hand[tip] = Landmark(0.5, 0.35 if is_extended else 0.7)
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

