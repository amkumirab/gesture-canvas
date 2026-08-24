# Gesture Canvas

A real-time hand-tracking canvas for drawing, moving, and extruding shapes using Python, OpenCV, and MediaPipe.

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
- Scale and rotate a grabbed drawing by pinching it with both hands
- Smoothly return to one-hand movement when either hand releases
- Adaptive cursor smoothing and short dropout recovery for continuous strokes
- Adjustable 2-30 px brush with an accurate on-screen size preview
- Toggleable in-app controls guide
- Convert a filled closed drawing into a shaded 3D extrusion
- Move 3D objects with one pinch and scale, spin, or tilt them with two hands
- Adjust extrusion depth and keep every 3D edit in undo/redo history
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
| Scale and rotate | Keep the shape pinched, pinch with the second hand, then change hand distance and angle |
| Create a 3D extrusion | Point inside a filled closed shape and press `E` |
| Manipulate a 3D object | Drag with one pinch; use two pinches to scale, spin, and tilt |
| Decrease / increase 3D depth | `-` / `+` |
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
  spatial.py      3D extrusion, projection, shading, and manipulation
  gestures.py     deterministic hand-pose rules
  landmarks.py    feature extraction and coordinate conversion
tests/            fast unit tests that do not require a webcam
```

## Run tests

```bash
pytest
```

## Current scope

Version `0.1.0` recognizes static hand poses and supports lightweight 3D extrusion. The 3D workspace currently creates independent extruded objects; drawing directly on a rotated object's surface is a future milestone.

## Privacy

All webcam processing happens locally. No frame is uploaded to a server.

## License

MIT
