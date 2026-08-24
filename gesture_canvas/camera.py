"""Shared webcam and MediaPipe Tasks helpers."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2

from .landmarks import Landmark, as_landmarks


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True, slots=True)
class DetectedHand:
    landmarks: list[Landmark]
    handedness: str
    confidence: float


def default_landmarker_path() -> Path:
    """Use a per-user cache so installed packages remain read-only."""

    return Path.home() / ".cache" / "gesture-canvas" / "hand_landmarker.task"


def ensure_landmarker_asset(path: Path | None = None) -> Path:
    """Download the official MediaPipe landmarker asset with an atomic rename."""

    destination = path or default_landmarker_path()
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, temporary)
        temporary.replace(destination)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe Hand Landmarker asset. Check your "
            "internet connection and try again."
        ) from exc
    return destination


class HandTracker:
    """Small adapter around MediaPipe's video-mode Hand Landmarker."""

    def __init__(self, asset_path: Path | None = None) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe could not be imported. Install project dependencies first."
            ) from exc

        asset = ensure_landmarker_asset(asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(asset)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        self._last_timestamp = 0

    def detect_all(self, frame) -> list[DetectedHand]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(int(time.monotonic() * 1000), self._last_timestamp + 1)
        self._last_timestamp = timestamp
        result = self._landmarker.detect_for_video(image, timestamp)
        if not result.hand_landmarks:
            return []

        detected: list[DetectedHand] = []
        for index, raw_landmarks in enumerate(result.hand_landmarks):
            handedness = "unknown"
            confidence = 0.0
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness = str(category.category_name).lower()
                confidence = float(category.score)
            detected.append(
                DetectedHand(
                    landmarks=as_landmarks(raw_landmarks),
                    handedness=handedness,
                    confidence=confidence,
                )
            )
        return detected

    def detect(self, frame):
        hands = self.detect_all(frame)
        return hands[0].landmarks if hands else None

    def close(self) -> None:
        self._landmarker.close()


def create_hand_tracker(asset_path: Path | None = None) -> HandTracker:
    return HandTracker(asset_path)


def open_camera(index: int, width: int = 1280, height: int = 720):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Check camera permissions or try --camera 1."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def detect_hands(tracker, frame) -> list[DetectedHand]:
    return tracker.detect_all(frame)
