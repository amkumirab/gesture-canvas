"""Collect normalized hand landmarks for a user-defined gesture label."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from .camera import create_hand_tracker, detect_landmarks, open_camera
from .gestures import Gesture
from .landmarks import normalized_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a personal gesture dataset.")
    parser.add_argument("--label", required=True, choices=[g.value for g in Gesture])
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/gestures.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    camera = open_camera(args.camera)
    try:
        tracker = create_hand_tracker()
    except Exception:
        camera.release()
        raise
    collected = 0
    last_capture = 0.0
    fieldnames = ["label", *[f"f{i}" for i in range(63)]]
    write_header = not args.output.exists() or args.output.stat().st_size == 0

    try:
        with args.output.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            while collected < args.samples:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The webcam stopped returning frames")
                frame = cv2.flip(frame, 1)
                landmarks = detect_landmarks(tracker, frame)
                now = time.monotonic()
                if landmarks is not None and now - last_capture >= 0.05:
                    features = normalized_features(landmarks)
                    row = {"label": args.label}
                    row.update({f"f{i}": float(value) for i, value in enumerate(features)})
                    writer.writerow(row)
                    collected += 1
                    last_capture = now

                cv2.putText(frame, f"Label: {args.label}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 255), 2)
                cv2.putText(frame, f"Samples: {collected}/{args.samples}", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(frame, "Move your hand slightly; Q cancels", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Gesture Dataset Collector", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    print(f"Collected {collected} '{args.label}' samples in {args.output}")


if __name__ == "__main__":
    main()
