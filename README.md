# Gesture Canvas

A real-time hand-tracking canvas for drawing and arranging shapes in 3D space using Python, OpenCV, and MediaPipe.

The project works immediately with deterministic hand-geometry rules and processes every camera frame locally.

## Input and output

**Input:** live webcam frames and the 21 hand landmarks detected by MediaPipe.

**Output:** a low-latency drawing canvas controlled without a mouse, plus PNG images and MP4 session recordings.

## Features

- Guided two-step gesture calibration with live light and hand-quality feedback
- Persistent per-user pinch, smoothing, depth, camera, and selection settings
- Save and reopen editable `.gcanvas` projects with drawing and 3D state intact
- Preserve undo and redo history inside project files
- Recover the most recent drawing session with background autosave
- Manage 3D layers from an on-canvas panel with selection, visibility, locking, duplication, deletion, and stacking controls
- Record the complete camera interface or a clean canvas-only MP4 video
- Draw with one raised index finger
- Automatically fill a completed closed shape with its stroke color
- Erase with two raised fingers
- Point-and-hold toolbar selection with visible progress feedback
- Grab and move any connected drawing directly with a pinch-and-drag gesture
- Stable pinch-center control with short hand-tracking dropout recovery
- Scale and rotate a grabbed drawing by pinching it with both hands
- Intent-locked 3D transforms that isolate scale, spin, and tilt
- Explicit scale and X/Y/Z rotation controls with number shortcuts and a persistent mode indicator
- Stable two-hand transform arming with multi-frame intent confirmation
- Continuous rotation when hand labels briefly swap during tracking
- Smoothly return to one-hand movement when either hand releases
- Adaptive cursor smoothing and short dropout recovery for continuous strokes
- Adjustable 2-30 px brush with an accurate on-screen size preview
- Toggleable in-app controls guide
- Convert a filled closed drawing into a flat layer in 3D space
- Move layers along X/Y by dragging and along Z by moving your hand closer or farther
- Scale, spin, or tilt 3D layers with two hands
- Perspective sizing, front-to-back ordering, and undo/redo for every spatial edit
- Projected XYZ orientation gizmo and a live near-to-far depth ruler
- Undo, redo, clear, and PNG export
- Confidence and FPS display
- Unit tests for gesture logic, smoothing, 3D projection, and canvas history
- GitHub Actions tests on Python 3.10, 3.11, and 3.12

## Quick start

Python 3.10, 3.11, or 3.12 is recommended.

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
gesture-canvas
```

To open an existing editable project at launch:

```bash
gesture-canvas --project path/to/drawing.gcanvas
```

The previous drawing session is restored automatically. To deliberately start with a blank canvas instead:

```bash
gesture-canvas --no-restore
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
gesture-canvas
```

Allow camera access when your operating system asks for permission. If the wrong camera opens, run `gesture-canvas --camera 1`. A successful camera override is remembered for later launches.

Press `K` after launch to calibrate the app for your hand and camera. First hold thumb and index open, then follow the prompt and pinch them together. No calibration is required to start; validated defaults are always available.

On the first launch, the application downloads MediaPipe's official Hand Landmarker asset (about 8 MB) and keeps it in a local cache. Later launches work without another download.

## Controls

| Action | Hand gesture / keyboard |
|---|---|
| Start or repeat calibration | `K` |
| Restore default gesture settings | `D` |
| Save editable project | `Ctrl+S` |
| Open editable project | `Ctrl+O` |
| Start without restoring the previous session | Launch with `--no-restore` |
| Draw | Raise only the index finger |
| Auto-fill | Complete and close a shape, then lift or bend the drawing finger |
| Temporary eraser | Raise index and middle fingers |
| Toolbar selection | Point with your index finger and hold over a button for 0.7 seconds |
| Move a drawing | Pinch thumb and index over a painted shape, drag, then open to drop |
| Scale | Keep two pinches held and move the hands apart or together |
| Spin | Turn the imaginary line between the two pinches like a steering wheel |
| Tilt | Move both pinches together, or move one hand closer than the other |
| Switch transform control | Release only the second pinch, then pinch again |
| Select 3D transform mode | `0` auto, `1` scale, `2` X tilt, `3` Y tilt, `4` Z spin |
| Reset 3D rotation | `R` |
| Create a 3D layer | Point inside a filled closed shape and press `E` |
| Open layer panel | `P`, then point at a layer or action and hold |
| Manage selected layer | Show/hide, lock/unlock, copy, delete, raise, or lower from the panel |
| Move along X/Y | Pinch the layer and drag left, right, up, or down |
| Move along Z | Keep pinching and move your hand closer to or farther from the camera |
| Start a two-hand transform | Hold both pinches steady until `TWO-HAND READY`, then move clearly |
| Rotate a 3D layer | Use two pinches; the live transform panel shows scale, X/Y tilt, and Z spin |
| Move farther / closer by keyboard | `-` / `+` |
| XYZ and depth guides | Displayed automatically while a 3D layer is selected |
| Export PNG | Toolbar `SAVE` or `S` |
| Start / stop MP4 recording | `F9` or `V` |
| Switch recording mode | `M` while idle: camera interface or clean canvas |
| Undo / Redo | Toolbar buttons or `Z` / `Y` |
| Clear | Toolbar `CLEAR` or `C` |
| Thinner / thicker brush | `[` / `]` |
| Show / hide controls guide | `H` |
| Quit | `Q` or `Esc` |

## Explicit 3D transform controls

Create a 3D layer with `E`, then select a control using the number keys. Hold two pinches steady to start manipulating the layer:

| Key | Control | Movement |
|---|---|---|
| `0` | Automatic | The existing intent detector chooses a control from your motion |
| `1` | Scale only | Move the pinches apart or together |
| `2` | X tilt only | Move both pinches up or down together |
| `3` | Y tilt only | Move both pinches left or right together, or change their relative hand depth |
| `4` | Z spin only | Turn the line between the pinches like a steering wheel |

Manual controls bypass intent detection and leave the other rotation axes, size, and position unchanged during two-hand transforms. A small indicator above the bottom status bar shows the selected control even between gestures. The projected axis guide highlights the selected rotation axis.

You can switch controls while holding both pinches. The layer keeps its current orientation and size, and the next frame establishes a new gesture reference. Choosing the same control again does not interrupt a gesture. Releasing one hand still returns to one-hand X/Y/Z movement. Number-key controls apply only to 3D layers; ordinary 2D drawings retain their existing two-hand behavior.

The selected control is a session preference, not drawing content: it survives Undo/Redo and opening a project during the session, but each new application launch starts in automatic mode.

## Project structure

```text
gesture_canvas/
  app.py          real-time application loop
  calibration.py  guided sampling and user-specific parameter calculation
  settings.py     validated JSON settings and local persistence
  project_file.py versioned editable project save/load and validation
  recovery.py     background session autosave coordination
  layer_panel.py  pointing-based 3D layer management panel
  recording.py    safe camera and clean-canvas MP4 recording
  canvas.py       drawing, compositing, undo/redo, export
  interaction.py  one- and two-hand manipulation coordination
  spatial.py      3D position, perspective projection, rotation, and depth ordering
  spatial_guides.py  projected XYZ gizmo and near-to-far depth ruler
  gestures.py     deterministic hand-pose rules
  landmarks.py    feature extraction and coordinate conversion
tests/            fast unit tests that do not require a webcam
```

## Run tests

```bash
pytest
```

## Current scope

Version `0.8.0` adds explicit 3D scale and rotation controls. Use `0` for automatic detection, `1` for scale, `2` for X tilt, `3` for Y tilt, and `4` for Z spin. Controls can be changed during a two-hand grab without resetting the layer's orientation or size.

MP4 recording remains available with `F9` or `V` to start and stop. Press `M` before recording to switch between the complete camera interface and a clean canvas-only video. The recording timer and active mode remain visible while capturing.

Recordings are finalized when recording stops or the application closes. Empty or failed recordings are removed instead of leaving unusable video files. Use `--record-mode canvas` to start directly in clean-canvas mode.

Frames are sampled using elapsed time: slower processing repeats the previous frame and faster processing drops excess frames, keeping playback at the captured speed. Videos have no audio and are saved under `outputs` by default; use `--output path/to/folder` to choose another folder.

Layer visibility, lock state, order, and selection are preserved in editable projects and automatic session recovery. Locked layers cannot be grabbed or transformed until they are unlocked from the panel.

Background recovery intentionally stores only the current state to keep frame interruptions minimal. Manually saved `.gcanvas` projects continue to preserve the complete undo and redo history.

A `.gcanvas` file preserves the 2D canvas, 3D layers, transforms, selected layer, brush settings, and undo/redo history. Projects created at a different camera resolution are fitted to the active canvas when opened.

Project saving is atomic: the existing file is replaced only after the new archive is complete. Project files contain drawing data and numeric workspace state; they never contain camera frames or executable content.

## Privacy

All webcam processing happens locally. No frame is uploaded to a server. Camera-mode recordings contain the visible camera view and are written only to the selected local output folder. Calibration stores only numeric settings, never camera frames. Session recovery contains only the drawing canvas and workspace state. On Windows, settings and recovery data are saved under `%LOCALAPPDATA%\gesture-canvas`.

## License

MIT
