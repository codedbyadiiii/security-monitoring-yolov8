from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from monitor import parse_polygon, process_video, threshold_preview


st.set_page_config(page_title="Security Monitor", page_icon="◈", layout="wide")
st.title("Security Monitor")
st.caption("Upload a video, define a zone, and turn tracked movement into auditable events.")

with st.sidebar:
    st.header("Detection settings")
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    confidence = st.slider("Confidence", 0.1, 0.9, 0.35, 0.05)
    confirmation_frames = st.slider("Confirmation frames", 1, 30, 5)
    st.markdown("**Zone points**")
    st.caption("Normalized coordinates: 0,0 is top-left and 1,1 is bottom-right.")
    polygon_text = st.text_input("Polygon", "0.15,0.15 0.85,0.15 0.85,0.85 0.15,0.85")

uploaded = st.file_uploader("Upload an MP4, MOV, AVI, or MKV video", type=["mp4", "mov", "avi", "mkv"])
if not uploaded:
    st.info("Upload a video to start monitoring.")
    st.stop()

try:
    polygon = parse_polygon(polygon_text)
except ValueError as error:
    st.error(str(error))
    st.stop()

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    input_path = temp_path / uploaded.name
    output_path = temp_path / "annotated_security_video.mp4"
    csv_path = temp_path / "security_events.csv"
    input_path.write_bytes(uploaded.getbuffer())

    st.subheader("Uploaded video")
    st.video(uploaded)
    st.divider()
    st.subheader("Classical segmentation preview")
    mode = st.radio("Thresholding mode", ["Binary", "Adaptive", "Otsu"], horizontal=True)
    if st.button("Preview first frame"):
        st.image(Image.fromarray(threshold_preview(input_path, mode)), caption=f"{mode} threshold")

    if st.button("Analyze video", type="primary"):
        progress = st.progress(0.0, text="Loading YOLO model...")
        events = process_video(input_path, output_path, csv_path, polygon, model_name, confidence, confirmation_frames, progress.progress)
        progress.progress(1.0, text="Analysis complete")
        st.session_state["events"] = events
        st.session_state["output_bytes"] = output_path.read_bytes()
        st.session_state["csv_bytes"] = csv_path.read_bytes()

if "events" in st.session_state:
    events = st.session_state["events"]
    st.subheader("Results")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Events", len(events))
    metric_columns[1].metric("Entries", sum(event["event"] == "entry" for event in events))
    metric_columns[2].metric("Exits", sum(event["event"] == "exit" for event in events))
    left, right = st.columns([2, 1])
    with left:
        st.video(st.session_state["output_bytes"])
    with right:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
        st.download_button("Download annotated video", st.session_state["output_bytes"], "annotated_security_video.mp4", "video/mp4")
        st.download_button("Download event CSV", st.session_state["csv_bytes"], "security_events.csv", "text/csv")
