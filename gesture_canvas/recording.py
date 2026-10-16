"""Safe MP4 recording for camera and clean-canvas sessions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Literal, Protocol

import cv2
import numpy as np


RecordingMode = Literal["camera", "canvas"]
VALID_RECORDING_MODES = frozenset({"camera", "canvas"})


class VideoWriter(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


WriterFactory = Callable[[str, int, float, tuple[int, int]], VideoWriter]
OutputValidator = Callable[[Path], bool]


class SessionRecorder:
    """Own one video writer and finalize or discard its output safely."""

    def __init__(
        self,
        output_dir: Path,
        fps: float = 30.0,
        mode: RecordingMode = "camera",
        writer_factory: WriterFactory | None = None,
        output_validator: OutputValidator | None = None,
        clock: Callable[[], float] = monotonic,
        timestamp: Callable[[], datetime] = datetime.now,
    ) -> None:
        if not 1.0 <= fps <= 240.0:
            raise ValueError("Recording FPS must be between 1 and 240")
        if mode not in VALID_RECORDING_MODES:
            raise ValueError("Unknown recording mode")
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.mode: RecordingMode = mode
        self._writer_factory = writer_factory or cv2.VideoWriter
        self._output_validator = output_validator or _contains_video_frame
        self._clock = clock
        self._timestamp = timestamp
        self._writer: VideoWriter | None = None
        self._path: Path | None = None
        self._frame_size: tuple[int, int] | None = None
        self._started_at: float | None = None
        self._first_frame_at: float | None = None
        self._last_frame: np.ndarray | None = None
        self._frame_count = 0

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def current_path(self) -> Path | None:
        return self._path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def set_mode(self, mode: RecordingMode) -> None:
        if mode not in VALID_RECORDING_MODES:
            raise ValueError("Unknown recording mode")
        if self.is_recording:
            raise RuntimeError("Recording mode cannot change while recording")
        self.mode = mode

    def toggle_mode(self) -> RecordingMode:
        mode: RecordingMode = "canvas" if self.mode == "camera" else "camera"
        self.set_mode(mode)
        return mode

    def start(self, frame_size: tuple[int, int]) -> Path:
        if self.is_recording:
            raise RuntimeError("A recording is already active")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Recording dimensions must be positive")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self._available_path()
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        try:
            writer = self._writer_factory(str(path), codec, self.fps, frame_size)
            opened = writer.isOpened()
        except Exception as error:
            path.unlink(missing_ok=True)
            raise OSError("Could not create the MP4 video writer") from error
        if not opened:
            try:
                writer.release()
            except Exception:
                pass
            path.unlink(missing_ok=True)
            raise OSError("Could not open the MP4 video writer")

        self._writer = writer
        self._path = path
        self._frame_size = frame_size
        self._started_at = self._clock()
        self._frame_count = 0
        return path

    def write(self, frame: np.ndarray) -> bool:
        """Sample frames on the output clock so playback retains capture speed."""

        if self._writer is None or self._frame_size is None:
            return False
        width, height = self._frame_size
        if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            raise ValueError("Recording frame dimensions or data type changed")
        now = self._clock()
        if self._first_frame_at is None:
            self._first_frame_at = now
        elapsed = max(0.0, now - self._first_frame_at)
        target_count = int(elapsed * self.fps + 1e-7) + 1
        if target_count <= self._frame_count:
            return False
        try:
            while self._last_frame is not None and self._frame_count < target_count - 1:
                self._writer.write(self._last_frame)
                self._frame_count += 1
            captured = np.ascontiguousarray(frame).copy()
            self._writer.write(captured)
            self._last_frame = captured
            self._frame_count += 1
        except Exception as error:
            self._discard()
            raise OSError("Could not write the video frame") from error
        return True

    def elapsed(self, now: float | None = None) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, (self._clock() if now is None else now) - self._started_at)

    def stop(self) -> Path | None:
        if self._writer is None:
            return None
        writer = self._writer
        path = self._path
        frame_count = self._frame_count
        self._reset()
        try:
            writer.release()
        except Exception as error:
            if path is not None:
                path.unlink(missing_ok=True)
            raise OSError("Could not finalize the video recording") from error

        if path is None or frame_count == 0:
            if path is not None:
                path.unlink(missing_ok=True)
            return None
        if not path.is_file() or path.stat().st_size == 0:
            path.unlink(missing_ok=True)
            raise OSError("The video writer produced an empty recording")
        try:
            valid_output = self._output_validator(path)
        except Exception:
            valid_output = False
        if not valid_output:
            path.unlink(missing_ok=True)
            raise OSError("The finalized recording could not be reopened")
        return path

    def close(self) -> Path | None:
        return self.stop()

    def _available_path(self) -> Path:
        stem = f"gesture_canvas_{self._timestamp():%Y%m%d_%H%M%S}_{self.mode}"
        path = self.output_dir / f"{stem}.mp4"
        suffix = 2
        while path.exists():
            path = self.output_dir / f"{stem}_{suffix}.mp4"
            suffix += 1
        return path

    def _discard(self) -> None:
        writer = self._writer
        path = self._path
        self._reset()
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if path is not None:
            path.unlink(missing_ok=True)

    def _reset(self) -> None:
        self._writer = None
        self._path = None
        self._frame_size = None
        self._started_at = None
        self._first_frame_at = None
        self._last_frame = None
        self._frame_count = 0


def _contains_video_frame(path: Path) -> bool:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        capture.release()


def draw_recording_hud(
    frame: np.ndarray,
    elapsed_seconds: float,
    mode: RecordingMode,
) -> None:
    """Draw a compact recording timer in the lower-right corner."""

    if mode not in VALID_RECORDING_MODES:
        raise ValueError("Unknown recording mode")
    height, width = frame.shape[:2]
    if height < 80 or width < 220:
        return
    elapsed = max(0, int(elapsed_seconds))
    minutes, seconds = divmod(elapsed, 60)
    label = f"REC {minutes:02d}:{seconds:02d}  {mode.upper()}"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    right = width - 12
    left = max(8, right - text_size[0] - 38)
    bottom = height - 34
    top = bottom - 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.circle(frame, (left + 14, top + 15), 6, (45, 55, 245), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (left + 28, top + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )
