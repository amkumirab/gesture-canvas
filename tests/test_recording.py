from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gesture_canvas.recording import SessionRecorder, draw_recording_hud


class FakeWriter:
    def __init__(self, path: str, opened: bool = True, fail_write: bool = False):
        self.path = Path(path)
        self.opened = opened
        self.fail_write = fail_write
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        if self.fail_write:
            raise RuntimeError("write failed")
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True
        if self.opened:
            self.path.write_bytes(b"video" if self.frames else b"")


class WriterFactory:
    def __init__(self, opened: bool = True, fail_write: bool = False):
        self.opened = opened
        self.fail_write = fail_write
        self.writers: list[FakeWriter] = []

    def __call__(self, path, _codec, _fps, _size):
        writer = FakeWriter(path, self.opened, self.fail_write)
        self.writers.append(writer)
        return writer


def fixed_timestamp() -> datetime:
    return datetime(2026, 9, 29, 14, 5, 7)


def test_recording_writes_frames_and_finalizes_mp4(tmp_path: Path):
    factory = WriterFactory()
    times = iter((10.0, 10.0, 12.5))
    recorder = SessionRecorder(
        tmp_path,
        fps=24,
        writer_factory=factory,
        output_validator=lambda _path: True,
        clock=lambda: next(times),
        timestamp=fixed_timestamp,
    )

    path = recorder.start((64, 48))
    assert path.name == "gesture_canvas_20260929_140507_camera.mp4"
    assert recorder.is_recording
    assert recorder.write(np.zeros((48, 64, 3), dtype=np.uint8))
    assert recorder.elapsed() == 2.5
    assert recorder.stop() == path

    assert path.read_bytes() == b"video"
    assert len(factory.writers[0].frames) == 1
    assert factory.writers[0].released
    assert not recorder.is_recording


def test_empty_recording_is_removed_and_close_is_safe(tmp_path: Path):
    recorder = SessionRecorder(
        tmp_path,
        writer_factory=WriterFactory(),
        output_validator=lambda _path: True,
        timestamp=fixed_timestamp,
    )
    path = recorder.start((32, 24))

    assert recorder.stop() is None
    assert not path.exists()
    assert recorder.close() is None


def test_recording_uses_a_unique_name_and_toggles_mode(tmp_path: Path):
    existing = tmp_path / "gesture_canvas_20260929_140507_canvas.mp4"
    existing.write_bytes(b"old")
    recorder = SessionRecorder(
        tmp_path,
        mode="canvas",
        writer_factory=WriterFactory(),
        output_validator=lambda _path: True,
        timestamp=fixed_timestamp,
    )

    path = recorder.start((32, 24))
    assert path.name.endswith("_canvas_2.mp4")
    recorder.write(np.zeros((24, 32, 3), dtype=np.uint8))
    recorder.stop()
    assert recorder.toggle_mode() == "camera"


def test_unavailable_writer_and_failed_frame_leave_no_file(tmp_path: Path):
    unavailable = SessionRecorder(
        tmp_path,
        writer_factory=WriterFactory(opened=False),
        output_validator=lambda _path: True,
        timestamp=fixed_timestamp,
    )
    with pytest.raises(OSError, match="open"):
        unavailable.start((32, 24))

    failing = SessionRecorder(
        tmp_path,
        writer_factory=WriterFactory(fail_write=True),
        output_validator=lambda _path: True,
        timestamp=fixed_timestamp,
    )
    path = failing.start((32, 24))
    with pytest.raises(OSError, match="frame"):
        failing.write(np.zeros((24, 32, 3), dtype=np.uint8))
    assert not path.exists()
    assert not failing.is_recording


def test_writer_construction_failure_is_reported_as_output_error(tmp_path: Path):
    def fail_factory(*_args):
        raise RuntimeError("codec failure")

    recorder = SessionRecorder(tmp_path, writer_factory=fail_factory)
    with pytest.raises(OSError, match="create"):
        recorder.start((32, 24))


def test_recorder_validates_settings_and_frame_shape(tmp_path: Path):
    with pytest.raises(ValueError, match="FPS"):
        SessionRecorder(tmp_path, fps=0)
    with pytest.raises(ValueError, match="mode"):
        SessionRecorder(tmp_path, mode="unknown")

    recorder = SessionRecorder(
        tmp_path,
        writer_factory=WriterFactory(),
        output_validator=lambda _path: True,
    )
    with pytest.raises(ValueError, match="dimensions"):
        recorder.start((0, 24))
    recorder.start((32, 24))
    with pytest.raises(ValueError, match="dimensions"):
        recorder.write(np.zeros((20, 32, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="mode"):
        recorder.toggle_mode()
    recorder.stop()


def test_recording_hud_draws_timer_and_validates_mode():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    draw_recording_hud(frame, 65.9, "camera")
    assert np.count_nonzero(frame) > 0
    with pytest.raises(ValueError, match="mode"):
        draw_recording_hud(frame, 0, "unknown")


def test_invalid_finalized_video_is_removed(tmp_path: Path):
    recorder = SessionRecorder(
        tmp_path,
        writer_factory=WriterFactory(),
        output_validator=lambda _path: False,
        timestamp=fixed_timestamp,
    )
    path = recorder.start((32, 24))
    recorder.write(np.zeros((24, 32, 3), dtype=np.uint8))

    with pytest.raises(OSError, match="reopened"):
        recorder.stop()
    assert not path.exists()


def test_recording_preserves_duration_when_capture_is_slower_than_output(tmp_path: Path):
    factory = WriterFactory()
    now = [0.0]
    recorder = SessionRecorder(
        tmp_path,
        fps=30,
        writer_factory=factory,
        output_validator=lambda _path: True,
        clock=lambda: now[0],
    )
    recorder.start((32, 24))
    for index in range(11):
        now[0] = index / 10
        recorder.write(np.full((24, 32, 3), index, dtype=np.uint8))
    recorder.stop()

    frames = factory.writers[0].frames
    assert len(frames) == 31
    np.testing.assert_array_equal(frames[0], 0)
    np.testing.assert_array_equal(frames[1], 0)
    np.testing.assert_array_equal(frames[-1], 10)


def test_recording_drops_excess_frames_when_capture_is_faster_than_output(tmp_path: Path):
    factory = WriterFactory()
    now = [0.0]
    recorder = SessionRecorder(
        tmp_path,
        fps=10,
        writer_factory=factory,
        output_validator=lambda _path: True,
        clock=lambda: now[0],
    )
    recorder.start((32, 24))
    for index in range(31):
        now[0] = index / 30
        recorder.write(np.full((24, 32, 3), index, dtype=np.uint8))
    recorder.stop()

    frames = factory.writers[0].frames
    assert len(frames) == 11
    np.testing.assert_array_equal(frames[-1], 30)
