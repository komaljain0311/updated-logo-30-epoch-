
# 🔍 YOLOv8 Logo Detection Streamlit App

## 📘 Overview
This is a Streamlit-based application for detecting logos in images and videos using a custom-trained YOLOv8 model. The app enables users to upload files or select from a local media library. Detected logos are displayed with bounding boxes and their occurrence frequencies.

---

## 🧠 Features


## model train using google colab 
(T4-gpu) 
upload google colab notebbok on gihub (Komal(2)1.ipynb)

## Roboflow dataset link
license: CC BY 4.0
  project: logo-detection-wrdch
  url: https://universe.roboflow.com/testspace-6ia9d/logo-detection-wrdch/dataset/1
  version: 1
  workspace: testspace-6ia9d
test: ../test/images
train: ../train/images
val: ../valid/images


### ✅ Input Options
- **Upload Image**: Supports `.jpg`, `.jpeg`, `.png`
- **Upload Video**: Supports `.mp4`, `.mov`, `.avi`
- **Search from Library**: Select media from predefined folders

### ✅ Output
- Annotated images or video frames with detected logos
- Frequency count of detected logos using `collections.Counter`

---

## 🧾 Dependencies

- `streamlit`
- `PIL`
- `cv2` (OpenCV)
- `numpy`
- `tempfile`
- `ultralytics` (YOLOv8)
- `collections`

---

## 📁 Directory Structure

```
project-root/
│
├── app.py                      # Main Streamlit app
├── runs/detect/train/weights/best.pt   # YOLOv8 weights
├── logo-detection-1/
│   └── database/
│       ├── image.jpg/          # Sample images
│       └── video.mp4/          # Sample video


## 🧩 Future Enhancements
- Batch processing
- Save results to disk
- Add detection confidence filter
- Improve video playback performance

---

## 🛡️ Limitations
- Only supports fixed media directories
- No advanced error handling
- Video playback is not real-time optimized
