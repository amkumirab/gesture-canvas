"""Versioned, editable Gesture Canvas project files."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import cv2
import numpy as np

from .canvas import CanvasProjectState, CanvasSnapshot
from .controls import MAX_BRUSH_SIZE, MIN_BRUSH_SIZE
from .spatial import SpatialShape, SpatialSnapshot


PROJECT_FORMAT = "gesture-canvas-project"
PROJECT_VERSION = 1
PROJECT_EXTENSION = ".gcanvas"
MAX_CANVAS_DIMENSION = 8192
MAX_HISTORY_LIMIT = 100
MAX_SPATIAL_OBJECTS = 500
MAX_CONTOUR_POINTS = 20_000
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
VALID_TOOLS = frozenset({"blue", "green", "red", "black", "eraser"})


class ProjectFormatError(ValueError):
    """Raised when a project file is malformed or unsupported."""


@dataclass(slots=True)
class EditableProject:
    canvas: CanvasProjectState
    brush_size: int
    active_tool: str


def save_project(
    path: Path,
    canvas: CanvasProjectState,
    brush_size: int,
    active_tool: str,
) -> Path:
    """Atomically save an editable project without executable payloads."""

    path = _with_extension(Path(path))
    if not MIN_BRUSH_SIZE <= brush_size <= MAX_BRUSH_SIZE:
        raise ValueError("Brush size is outside the supported range")
    if active_tool not in VALID_TOOLS:
        raise ValueError("Unknown drawing tool")
    _validate_state_for_save(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshots = [canvas.current, *canvas.undo, *canvas.redo]
    current_index = 0
    undo_indices = list(range(1, 1 + len(canvas.undo)))
    redo_indices = list(range(1 + len(canvas.undo), len(snapshots)))
    manifest: dict[str, Any] = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "canvas": {
            "width": canvas.width,
            "height": canvas.height,
            "history_limit": canvas.history_limit,
        },
        "workspace": {"brush_size": brush_size, "active_tool": active_tool},
        "history": {
            "current": current_index,
            "undo": undo_indices,
            "redo": redo_indices,
        },
        "snapshots": [
            _snapshot_manifest(snapshot, index)
            for index, snapshot in enumerate(snapshots)
        ],
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with ZipFile(
            temporary_path,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            for index, snapshot in enumerate(snapshots):
                prefix = f"snapshots/{index:03d}"
                archive.writestr(
                    f"{prefix}/strokes.png",
                    _encode_png(snapshot.strokes),
                )
                archive.writestr(
                    f"{prefix}/mask.png",
                    _encode_png(snapshot.mask),
                )
        if temporary_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise OSError("Project file exceeds the supported size limit")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def load_project(
    path: Path,
    target_size: tuple[int, int] | None = None,
) -> EditableProject:
    """Load and validate a project, optionally fitting it to a new canvas size."""

    path = Path(path)
    try:
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ProjectFormatError("Project file is too large")
        with ZipFile(path, mode="r") as archive:
            _validate_archive(archive)
            manifest = _read_manifest(archive)
            project = _decode_project(archive, manifest)
    except ProjectFormatError:
        raise
    except (BadZipFile, KeyError, OSError, TypeError, ValueError) as error:
        raise ProjectFormatError("Could not read this project file") from error

    if target_size is not None:
        width, height = target_size
        if width <= 0 or height <= 0:
            raise ValueError("Target dimensions must be positive")
        if (width, height) != (project.canvas.width, project.canvas.height):
            project.canvas = _resize_state(project.canvas, width, height)
    return project


def _with_extension(path: Path) -> Path:
    return path if path.suffix.lower() == PROJECT_EXTENSION else path.with_suffix(PROJECT_EXTENSION)


def _validate_state_for_save(state: CanvasProjectState) -> None:
    if not 1 <= state.width <= MAX_CANVAS_DIMENSION:
        raise ValueError("Project width is outside the supported range")
    if not 1 <= state.height <= MAX_CANVAS_DIMENSION:
        raise ValueError("Project height is outside the supported range")
    if not 1 <= state.history_limit <= MAX_HISTORY_LIMIT:
        raise ValueError("Project history limit is outside the supported range")
    if (
        len(state.undo) > state.history_limit
        or len(state.redo) > state.history_limit
    ):
        raise ValueError("Project history exceeds its configured limit")
    for snapshot in (state.current, *state.undo, *state.redo):
        _validate_snapshot_arrays(snapshot, state.width, state.height)
        _validate_spatial_snapshot(snapshot.spatial)


def _snapshot_manifest(snapshot: CanvasSnapshot, index: int) -> dict[str, Any]:
    prefix = f"snapshots/{index:03d}"
    spatial = snapshot.spatial
    return {
        "strokes": f"{prefix}/strokes.png",
        "mask": f"{prefix}/mask.png",
        "selected_index": spatial.selected_index,
        "objects": [
            {
                "contour": shape.contour.astype(float).tolist(),
                "color": [int(channel) for channel in shape.color],
                "position": [float(value) for value in shape.position],
                "scale": float(shape.scale),
                "rotation": [
                    float(shape.rotation_x),
                    float(shape.rotation_y),
                    float(shape.rotation_z),
                ],
            }
            for shape in spatial.objects
        ],
    }


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("Could not encode project image data")
    return encoded.tobytes()


def _validate_archive(archive: ZipFile) -> None:
    entries = archive.infolist()
    if len(entries) > 2 * (2 * MAX_HISTORY_LIMIT + 1) + 1:
        raise ProjectFormatError("Project contains too many files")
    total_size = sum(entry.file_size for entry in entries)
    if total_size > MAX_ARCHIVE_BYTES:
        raise ProjectFormatError("Expanded project is too large")
    names = {entry.filename for entry in entries}
    if len(names) != len(entries):
        raise ProjectFormatError("Project contains duplicate files")
    if "manifest.json" not in names:
        raise ProjectFormatError("Project manifest is missing")


def _read_manifest(archive: ZipFile) -> dict[str, Any]:
    info = archive.getinfo("manifest.json")
    if info.file_size > 16 * 1024 * 1024:
        raise ProjectFormatError("Project manifest is too large")
    try:
        manifest = json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectFormatError("Project manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise ProjectFormatError("Project manifest must be an object")
    if manifest.get("format") != PROJECT_FORMAT:
        raise ProjectFormatError("This is not a Gesture Canvas project")
    if manifest.get("version") != PROJECT_VERSION:
        raise ProjectFormatError("This project version is not supported")
    return manifest


def _decode_project(archive: ZipFile, manifest: dict[str, Any]) -> EditableProject:
    canvas_data = _object(manifest.get("canvas"), "canvas")
    width = _integer(canvas_data.get("width"), "canvas width", 1, MAX_CANVAS_DIMENSION)
    height = _integer(canvas_data.get("height"), "canvas height", 1, MAX_CANVAS_DIMENSION)
    history_limit = _integer(
        canvas_data.get("history_limit"),
        "history limit",
        1,
        MAX_HISTORY_LIMIT,
    )
    workspace = _object(manifest.get("workspace"), "workspace")
    brush_size = _integer(
        workspace.get("brush_size"),
        "brush size",
        MIN_BRUSH_SIZE,
        MAX_BRUSH_SIZE,
    )
    active_tool = workspace.get("active_tool")
    if active_tool not in VALID_TOOLS:
        raise ProjectFormatError("Project drawing tool is invalid")

    snapshot_data = manifest.get("snapshots")
    if not isinstance(snapshot_data, list) or not snapshot_data:
        raise ProjectFormatError("Project snapshots are missing")
    if len(snapshot_data) > 2 * history_limit + 1:
        raise ProjectFormatError("Project contains too much history")
    snapshots = [
        _decode_snapshot(archive, item, width, height)
        for item in snapshot_data
    ]

    history = _object(manifest.get("history"), "history")
    current_index = _snapshot_index(history.get("current"), len(snapshots))
    undo_indices = _index_list(history.get("undo"), len(snapshots), history_limit)
    redo_indices = _index_list(history.get("redo"), len(snapshots), history_limit)
    all_indices = [current_index, *undo_indices, *redo_indices]
    if len(set(all_indices)) != len(all_indices) or len(all_indices) != len(snapshots):
        raise ProjectFormatError("Project history references are inconsistent")

    state = CanvasProjectState(
        width=width,
        height=height,
        history_limit=history_limit,
        current=snapshots[current_index],
        undo=[snapshots[index] for index in undo_indices],
        redo=[snapshots[index] for index in redo_indices],
    )
    return EditableProject(state, brush_size, active_tool)


def _decode_snapshot(
    archive: ZipFile,
    value: object,
    width: int,
    height: int,
) -> CanvasSnapshot:
    data = _object(value, "snapshot")
    strokes = _decode_png(archive, data.get("strokes"), color=True)
    mask = _decode_png(archive, data.get("mask"), color=False)
    if strokes.shape != (height, width, 3) or mask.shape != (height, width):
        raise ProjectFormatError("Project image dimensions are inconsistent")

    objects_data = data.get("objects")
    if not isinstance(objects_data, list) or len(objects_data) > MAX_SPATIAL_OBJECTS:
        raise ProjectFormatError("Project spatial objects are invalid")
    objects = [_decode_shape(item) for item in objects_data]
    selected_index = data.get("selected_index")
    if selected_index is not None:
        selected_index = _integer(
            selected_index,
            "selected object",
            0,
            max(len(objects) - 1, 0),
        )
        if not objects:
            raise ProjectFormatError("Selected object does not exist")
    return CanvasSnapshot(
        strokes=strokes,
        mask=mask,
        spatial=SpatialSnapshot(objects, selected_index),
    )


def _decode_png(archive: ZipFile, name: object, color: bool) -> np.ndarray:
    if not isinstance(name, str) or not name.startswith("snapshots/"):
        raise ProjectFormatError("Project image reference is invalid")
    try:
        payload = archive.read(name)
    except KeyError as error:
        raise ProjectFormatError("Project image data is missing") from error
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flag)
    if image is None or image.dtype != np.uint8:
        raise ProjectFormatError("Project image data is invalid")
    return image


def _decode_shape(value: object) -> SpatialShape:
    data = _object(value, "spatial object")
    contour_data = data.get("contour")
    if not isinstance(contour_data, list) or not 3 <= len(contour_data) <= MAX_CONTOUR_POINTS:
        raise ProjectFormatError("Spatial contour is invalid")
    contour = np.asarray(contour_data, dtype=np.float32)
    if contour.shape != (len(contour_data), 2) or not np.all(np.isfinite(contour)):
        raise ProjectFormatError("Spatial contour points are invalid")
    color = _number_tuple(data.get("color"), "spatial color", 3, integer=True)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ProjectFormatError("Spatial color is outside the supported range")
    position = _number_tuple(data.get("position"), "spatial position", 3)
    rotation = _number_tuple(data.get("rotation"), "spatial rotation", 3)
    scale = _finite_number(data.get("scale"), "spatial scale")
    if not 0.05 <= scale <= 20:
        raise ProjectFormatError("Spatial scale is outside the supported range")
    return SpatialShape(
        contour=contour,
        color=tuple(int(channel) for channel in color),
        position=tuple(float(item) for item in position),
        scale=scale,
        rotation_x=float(rotation[0]),
        rotation_y=float(rotation[1]),
        rotation_z=float(rotation[2]),
    )


def _resize_state(state: CanvasProjectState, width: int, height: int) -> CanvasProjectState:
    scale_x = width / state.width
    scale_y = height / state.height
    depth_scale = max(width, height) / max(state.width, state.height)

    def resize(snapshot: CanvasSnapshot) -> CanvasSnapshot:
        strokes = cv2.resize(snapshot.strokes, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(snapshot.mask, (width, height), interpolation=cv2.INTER_LINEAR)
        objects = []
        for shape in snapshot.spatial.objects:
            resized = shape.copy()
            resized.contour[:, 0] *= scale_x
            resized.contour[:, 1] *= scale_y
            resized.position = (
                shape.position[0] * scale_x,
                shape.position[1] * scale_y,
                shape.position[2] * depth_scale,
            )
            objects.append(resized)
        return CanvasSnapshot(
            strokes=strokes,
            mask=mask,
            spatial=SpatialSnapshot(objects, snapshot.spatial.selected_index),
        )

    return CanvasProjectState(
        width=width,
        height=height,
        history_limit=state.history_limit,
        current=resize(state.current),
        undo=[resize(snapshot) for snapshot in state.undo],
        redo=[resize(snapshot) for snapshot in state.redo],
    )


def _validate_snapshot_arrays(snapshot: CanvasSnapshot, width: int, height: int) -> None:
    if snapshot.strokes.dtype != np.uint8 or snapshot.strokes.shape != (height, width, 3):
        raise ValueError("Project strokes have invalid dimensions or data type")
    if snapshot.mask.dtype != np.uint8 or snapshot.mask.shape != (height, width):
        raise ValueError("Project mask has invalid dimensions or data type")


def _validate_spatial_snapshot(snapshot: SpatialSnapshot) -> None:
    if len(snapshot.objects) > MAX_SPATIAL_OBJECTS:
        raise ValueError("Project contains too many spatial objects")
    if (
        snapshot.selected_index is not None
        and not 0 <= snapshot.selected_index < len(snapshot.objects)
    ):
        raise ValueError("Project selected object is invalid")
    for shape in snapshot.objects:
        if shape.contour.shape != (len(shape.contour), 2):
            raise ValueError("Project spatial contour is invalid")
        if not 3 <= len(shape.contour) <= MAX_CONTOUR_POINTS:
            raise ValueError("Project spatial contour size is invalid")
        numbers = (
            *shape.color,
            *shape.position,
            shape.scale,
            shape.rotation_x,
            shape.rotation_y,
            shape.rotation_z,
        )
        if not np.all(np.isfinite(shape.contour)) or not all(
            isfinite(float(item)) for item in numbers
        ):
            raise ValueError("Project spatial object contains invalid numbers")
        if any(not 0 <= int(channel) <= 255 for channel in shape.color):
            raise ValueError("Project spatial color is outside the supported range")
        if not 0.05 <= float(shape.scale) <= 20:
            raise ValueError("Project spatial scale is outside the supported range")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectFormatError(f"Project {label} is invalid")
    return value


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ProjectFormatError(f"Project {label} is invalid")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectFormatError(f"Project {label} is invalid")
    number = float(value)
    if not isfinite(number):
        raise ProjectFormatError(f"Project {label} is invalid")
    return number


def _number_tuple(
    value: object,
    label: str,
    length: int,
    integer: bool = False,
) -> tuple[float, ...] | tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProjectFormatError(f"Project {label} is invalid")
    if integer:
        return tuple(_integer(item, label, -1_000_000, 1_000_000) for item in value)
    return tuple(_finite_number(item, label) for item in value)


def _snapshot_index(value: object, count: int) -> int:
    return _integer(value, "snapshot index", 0, count - 1)


def _index_list(value: object, count: int, limit: int) -> list[int]:
    if not isinstance(value, list) or len(value) > limit:
        raise ProjectFormatError("Project history list is invalid")
    return [_snapshot_index(item, count) for item in value]
