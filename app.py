import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from collections import Counter

# Load your YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Update this if needed

# Set Streamlit config
st.set_page_config(page_title="YOLOv8 Logo Detection", layout="wide")

st.title("🔍 Logo Detection with YOLOv8")
st.markdown("Use the left panel to select an image/video and detect logos. Output and frequency count will be shown on the right.")

# Directory setup
PHOTO_FOLDER = "logo-detection-1/database/image.jpg"
VIDEO_FOLDER = "logo-detection-1/database/video.mp4"

# Create two columns
col1, col2 = st.columns([1, 2])  # Left: Input | Right: Output

with col1:
    st.header("📥 Input Options")
    input_type = st.radio("Choose Input Type", ["Upload Image", "Upload Video", "Search from Library"])

    selected_file = None

    if input_type == "Upload Image":
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            selected_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            selected_file.write(uploaded_img.read())

    elif input_type == "Upload Video":
        uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if uploaded_vid is not None:
            selected_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            selected_file.write(uploaded_vid.read())

    elif input_type == "Search from Library":
        media_type = st.selectbox("Select Media Type", ["Photo", "Video"])
        if media_type == "Photo":
            photo_files = os.listdir(PHOTO_FOLDER)
            selected_photo = st.selectbox("Choose an Image", photo_files)
            selected_file = os.path.join(PHOTO_FOLDER, selected_photo)
        elif media_type == "Video":
            video_files = os.listdir(VIDEO_FOLDER)
            selected_video = st.selectbox("Choose a Video", video_files)
            selected_file = os.path.join(VIDEO_FOLDER, selected_video)

    detect_btn = st.button("🚀 Detect Logos")

with col2:
    st.header("📤 Output")
    if detect_btn and selected_file:
        if input_type in ["Upload Image", "Search from Library"] and selected_file:
            results = model(selected_file.name if hasattr(selected_file, 'name') else selected_file)

            for r in results:
                annotated_img = r.plot()
                st.image(annotated_img, caption="🔎 Detected Logos", use_column_width=True)

                # Frequency count
                labels = r.names
                counts = Counter(r.boxes.cls.tolist())
                st.markdown("### 📊 Logo Detection Frequency")
                for cls_id, freq in counts.items():
                    st.write(f"{labels[int(cls_id)]}: {freq} times")

        elif input_type in ["Upload Video", "Search from Library"] and selected_file:
            stframe = st.empty()
            cap = cv2.VideoCapture(selected_file.name if hasattr(selected_file, 'name') else selected_file)
            full_counts = Counter()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                boxes = results[0].boxes
                for cls_id in boxes.cls.tolist():
                    full_counts[int(cls_id)] += 1

                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR")

            cap.release()

            st.markdown("### 📊 Logo Detection Frequency")
            for cls_id, freq in full_counts.items():
                st.write(f"{model.names[int(cls_id)]}: {freq} times")