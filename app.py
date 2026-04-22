import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 🔥 REAL FIX (must be here)
import subprocess
subprocess.run("pip uninstall -y opencv-python", shell=True)

import cv2
cv2.setNumThreads(0)

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image


# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Object Detection App", layout="centered")

st.title("🚀 Real-Time Object Detection")
st.write("Upload an image and detect objects using YOLOv8")

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ==============================
# Confidence Slider
# ==============================
confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# ==============================
# File Upload
# ==============================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width='stretch')


    img = np.array(image)

    # ==============================
    # Run Detection
    # ==============================
    results = model(img, conf=confidence)

    # ==============================
    # Show Output
    # ==============================
    for r in results:
        annotated_img = r.plot()

        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.image(annotated_img, caption="Detected Objects", width='stretch')


        # ==============================
        # Show Details
        # ==============================
        st.subheader("Detected Objects:")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            st.write(f"{label} ({conf:.2f})")
