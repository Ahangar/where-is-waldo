import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2


@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("YOLO Object Detection")

uploaded = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input")

    # Annotated image with bounding boxes

    results = model(image)
    annotated = results[0].plot()

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # convert to RGB

    # Display in Jupyter
    plt.figure(figsize=(10,10))
    plt.imshow(annotated)
    plt.axis("off")
    plt.show()

    st.image(annotated, caption="Detected")
