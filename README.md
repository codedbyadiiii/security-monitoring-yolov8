# Security Monitoring with YOLOv8 and ByteTrack

Upload a video, define a polygon security zone, track people with YOLOv8 + ByteTrack, and export stable entry/exit events.

## Run

```bash
cd security-monitoring-yolov8
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

The first analysis downloads the selected Ultralytics model. The app only processes uploaded video files; it does not send video to an external service.

## Zone format

Enter normalized polygon points in the sidebar, separated by spaces:

```text
0.15,0.15 0.85,0.15 0.85,0.85 0.15,0.85
```

Coordinates use the original video frame: `(0, 0)` is top-left and `(1, 1)` is bottom-right. A person is considered inside when their bounding-box center is inside the polygon for the configured number of consecutive frames. The same debounce applies to exits.

## Outputs

- Annotated MP4 with track IDs and zone state
- CSV containing UTC timestamp, video frame, track ID, and `entry`/`exit` event
- Binary, adaptive, and Otsu threshold preview of the first frame
