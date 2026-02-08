import streamlit as st
from PIL import Image
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("YOLO Object Detection")

uploaded = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input")

    results = model(image)
    annotated = results[0].plot()

    st.image(annotated, caption="Detected")
