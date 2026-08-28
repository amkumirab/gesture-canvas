# Gesture Canvas

A real-time hand-tracking canvas for drawing and arranging shapes in 3D space using Python, OpenCV, and MediaPipe.

The project works immediately with deterministic hand-geometry rules and processes every camera frame locally.

## Input and output

**Input:** live webcam frames and the 21 hand landmarks detected by MediaPipe.

**Output:** a low-latency drawing canvas controlled without a mouse, plus PNG exports on a clean white background.

## Features

- Draw with one raised index finger
- Automatically fill a completed closed shape with its stroke color
- Erase with two raised fingers
- Point-and-hold toolbar selection with visible progress feedback
- Grab and move any connected drawing directly with a pinch-and-drag gesture
- Stable pinch-center control with short hand-tracking dropout recovery
- Scale and rotate a grabbed drawing by pinching it with both hands
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

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
gesture-canvas
```

Allow camera access when your operating system asks for permission. If the wrong camera opens, run `gesture-canvas --camera 1`.

On the first launch, the application downloads MediaPipe's official Hand Landmarker asset (about 8 MB) and keeps it in a local cache. Later launches work without another download.

## Controls

| Action | Hand gesture / keyboard |
|---|---|
| Draw | Raise only the index finger |
| Auto-fill | Complete and close a shape, then lift or bend the drawing finger |
| Temporary eraser | Raise index and middle fingers |
| Toolbar selection | Point with your index finger and hold over a button for 0.7 seconds |
| Move a drawing | Pinch thumb and index over a painted shape, drag, then open to drop |
| Scale | Keep two pinches held and move the hands apart or together |
| Spin | Turn the imaginary line between the two pinches like a steering wheel |
| Tilt | Move both pinches together, or move one hand closer than the other |
| Create a 3D layer | Point inside a filled closed shape and press `E` |
| Move along X/Y | Pinch the layer and drag left, right, up, or down |
| Move along Z | Keep pinching and move your hand closer to or farther from the camera |
| Rotate a 3D layer | Use two pinches; the live transform panel shows scale, X/Y tilt, and Z spin |
| Move farther / closer by keyboard | `-` / `+` |
| XYZ and depth guides | Displayed automatically while a 3D layer is selected |
| Save | Toolbar `SAVE` or `S` |
| Undo / Redo | Toolbar buttons or `Z` / `Y` |
| Clear | Toolbar `CLEAR` or `C` |
| Thinner / thicker brush | `[` / `]` |
| Show / hide controls guide | `H` |
| Quit | `Q` or `Esc` |

## Project structure

```text
gesture_canvas/
  app.py          real-time application loop
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

Version `0.2.0` adds dropout-resistant manipulation, separate drawing and pinch smoothing, clearer live controls, and independent flat drawings in a perspective 3D workspace. Moving a hand toward or away from the camera controls the selected layer's Z position.

## Privacy

All webcam processing happens locally. No frame is uploaded to a server.

## License

MIT
