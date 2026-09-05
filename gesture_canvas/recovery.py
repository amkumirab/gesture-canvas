"""Non-blocking session autosave and recovery support."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .canvas import CanvasProjectState, DrawingCanvas
from .project_file import save_project


SaveFunction = Callable[[Path, CanvasProjectState, int, str], Path]


class AutosaveController:
    """Write detached project snapshots without blocking the camera loop."""

    def __init__(
        self,
        path: Path,
        interval_seconds: float = 8.0,
        save_function: SaveFunction = save_project,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Autosave interval must be positive")
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self._save_function = save_function
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gesture-canvas-autosave",
        )
        self._future: Future[Path] | None = None
        self._pending_signature: tuple[int, int, str] | None = None
        self._saved_signature: tuple[int, int, str] | None = None
        self._next_due: float | None = None
        self._error: str | None = None
        self._closed = False

    @property
    def saved_revision(self) -> int | None:
        self._collect_finished()
        return None if self._saved_signature is None else self._saved_signature[0]

    def maybe_save(
        self,
        canvas: DrawingCanvas,
        brush_size: int,
        active_tool: str,
        now: float,
    ) -> bool:
        """Schedule a save when due and return whether work was submitted."""

        if self._closed:
            raise RuntimeError("Autosave controller is closed")
        self._collect_finished()
        if self._next_due is None:
            self._next_due = now + self.interval_seconds
            return False
        if self._future is not None or now < self._next_due:
            return False
        self._next_due = now + self.interval_seconds
        signature = (canvas.revision, brush_size, active_tool)
        if signature == self._saved_signature:
            return False
        state = canvas.project_state(
            finalize_interactions=False,
            include_history=False,
        )
        self._pending_signature = signature
        self._future = self._executor.submit(
            self._save_function,
            self.path,
            state,
            brush_size,
            active_tool,
        )
        return True

    def save_now(
        self,
        canvas: DrawingCanvas,
        brush_size: int,
        active_tool: str,
    ) -> Path:
        """Synchronously capture the latest state, normally during shutdown."""

        if self._closed:
            raise RuntimeError("Autosave controller is closed")
        self.flush()
        signature = (canvas.revision, brush_size, active_tool)
        if signature == self._saved_signature and self.path.exists():
            return self.path
        state = canvas.project_state(
            finalize_interactions=False,
            include_history=False,
        )
        saved = self._save_function(
            self.path,
            state,
            brush_size,
            active_tool,
        )
        self._saved_signature = signature
        self._error = None
        return saved

    def consume_error(self) -> str | None:
        """Return the latest background error once without raising in the UI loop."""

        self._collect_finished()
        error = self._error
        self._error = None
        return error

    def flush(self) -> None:
        """Wait for pending work and record any error for later display."""

        if self._future is None:
            return
        future = self._future
        pending_signature = self._pending_signature
        self._future = None
        self._pending_signature = None
        try:
            future.result()
        except Exception as error:
            self._error = str(error) or type(error).__name__
        else:
            self._saved_signature = pending_signature

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._executor.shutdown(wait=True)
        self._closed = True

    def _collect_finished(self) -> None:
        if self._future is not None and self._future.done():
            self.flush()
