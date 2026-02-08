import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# Load custom YOLOv8 model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = Path("models/best.pt")
    assert model_path.exists(), f"Model not found: {model_path}"
    return YOLO(model_path)

model = load_model()


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("YOLOv8 Object Detection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Inference
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Run inference
    results = model(img_array, conf=0.1)

    # Draw results
    annotated_img = results[0].plot()  # returns BGR image
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Show output
    st.image(
        annotated_img,
        caption="Detection Result",
        use_column_width=True
    )
