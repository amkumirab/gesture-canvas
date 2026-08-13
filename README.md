# Gesture Canvas

A real-time hand-tracking drawing application built with Python, OpenCV, MediaPipe, and an optional user-trained PyTorch neural network.

The project works immediately with an explainable rule-based gesture baseline. It also includes a complete data collection and MLP training pipeline so the baseline can be replaced with a model trained on your own hand poses.

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
- Undo, redo, clear, and PNG export
- Confidence and FPS display
- Personal gesture dataset collector
- Optional PyTorch MLP training and confidence rejection
- Unit tests for gesture logic, features, smoothing, and canvas history
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

On the first launch, the application downloads MediaPipe's official Hand Landmarker model (about 8 MB) and keeps it in a local cache. Later launches work without another download.

## Controls

| Action | Hand gesture / keyboard |
|---|---|
| Draw | Raise only the index finger |
| Auto-fill | Complete and close a shape, then lift or bend the drawing finger |
| Temporary eraser | Raise index and middle fingers |
| Toolbar selection | Point with your index finger and hold over a button for 0.7 seconds |
| Move a drawing | Pinch thumb and index over a painted shape, drag, then open to drop |
| Scale and rotate | Keep the shape pinched, pinch with the second hand, then change hand distance and angle |
| Save | Toolbar `SAVE` or `S` |
| Undo / Redo | Toolbar buttons or `Z` / `Y` |
| Clear | Toolbar `CLEAR` or `C` |
| Quit | `Q` or `Esc` |

## Train your own gesture model

Install the ML extra:

```bash
pip install -e ".[ml,dev]"
```

Collect several slightly different samples for each label. A useful first dataset is 300 samples per class:

```bash
gesture-collect --label idle --samples 300
gesture-collect --label draw --samples 300
gesture-collect --label erase --samples 300
gesture-collect --label pinch --samples 300
gesture-collect --label open_palm --samples 300
```

Train and evaluate the MLP:

```bash
gesture-train --epochs 60
gesture-canvas --model models/gesture_mlp.pt
```

The raw camera images are not stored. The dataset contains only normalized `(x, y, z)` landmark features. Generated datasets, model weights, and drawings are ignored by Git by default.

## Project structure

```text
gesture_canvas/
  app.py          real-time application loop
  canvas.py       drawing, compositing, undo/redo, export
  interaction.py  one- and two-hand manipulation coordination
  gestures.py     explainable baseline classifier
  landmarks.py    feature extraction and coordinate conversion
  collect.py      personal dataset collection
  model.py        optional PyTorch MLP and inference
  train.py        training and holdout evaluation
tests/            fast unit tests that do not require a webcam
```

## Run tests

```bash
pytest
```

## Current scope

Version `0.1.0` recognizes static hand poses. A natural next milestone is a temporal 1D CNN or LSTM that receives a short sequence of landmarks and recognizes dynamic commands such as undo and redo.

## Privacy

All webcam processing happens locally. No frame is uploaded to a server.

## License

MIT
