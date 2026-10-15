from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

@dataclass
class TrackState:
    inside: bool = False
    inside_streak: int = 0
    outside_streak: int = 0


def parse_polygon(raw: str) -> list[tuple[float, float]]:
    """Parse normalized polygon points in the form x,y x,y ..."""
    points: list[tuple[float, float]] = []
    for item in raw.replace(";", " ").split():
        x_text, y_text = item.split(",")
        point = (float(x_text), float(y_text))
        if not 0 <= point[0] <= 1 or not 0 <= point[1] <= 1:
            raise ValueError("Polygon coordinates must be between 0 and 1.")
        points.append(point)
    if len(points) < 3:
        raise ValueError("A zone needs at least three points.")
    return points


def _inside(point: tuple[int, int], polygon: np.ndarray) -> bool:
    import cv2

    return cv2.pointPolygonTest(polygon, point, False) >= 0


def _draw_zone(frame: np.ndarray, polygon: np.ndarray) -> None:
    import cv2

    cv2.polylines(frame, [polygon], True, (0, 220, 255), 2)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (0, 180, 255))
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.putText(frame, "SECURITY ZONE", tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    csv_path: str | Path,
    polygon_points: Sequence[tuple[float, float]],
    model_name: str = "yolov8n.pt",
    confidence: float = 0.35,
    confirmation_frames: int = 5,
    progress_callback: Callable[[float], None] | None = None,
) -> list[dict[str, object]]:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the uploaded video.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    polygon = np.array([(int(x * width), int(y * height)) for x, y in polygon_points], dtype=np.int32)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError("OpenCV could not create the annotated output video.")

    model = YOLO(model_name)
    states: dict[int, TrackState] = {}
    events: list[dict[str, object]] = []
    frame_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=confidence, verbose=False)[0]
            _draw_zone(frame, polygon)
            boxes = result.boxes
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box.xyxy[0].int().cpu().tolist()
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                state = states.setdefault(track_id, TrackState())
                is_inside = _inside(center, polygon)
                if is_inside:
                    state.inside_streak += 1
                    state.outside_streak = 0
                else:
                    state.outside_streak += 1
                    state.inside_streak = 0

                event_type: str | None = None
                if not state.inside and state.inside_streak >= confirmation_frames:
                    state.inside = True
                    event_type = "entry"
                elif state.inside and state.outside_streak >= confirmation_frames:
                    state.inside = False
                    event_type = "exit"
                if event_type:
                    events.append({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "video_frame": frame_number,
                        "track_id": track_id,
                        "event": event_type,
                    })

                color = (0, 200, 80) if state.inside else (80, 170, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, center, 4, color, -1)
                cv2.putText(frame, f"person #{track_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            writer.write(frame)
            if progress_callback and total_frames:
                progress_callback(min(frame_number / total_frames, 1.0))
    finally:
        capture.release()
        writer.release()

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=["timestamp_utc", "video_frame", "track_id", "event"])
        writer_csv.writeheader()
        writer_csv.writerows(events)
    return events


def threshold_preview(input_path: str | Path, mode: str) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(input_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("Could not read a frame for threshold preview.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "Adaptive":
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif mode == "Otsu":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
