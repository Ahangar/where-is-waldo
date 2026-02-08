import streamlit as st
from PIL import Image
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont


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
    #annotated = results[0].plot()
    
    #bounding box
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
    scores = result.boxes.conf.cpu().numpy()  # confidence
    classes = result.boxes.cls.cpu().numpy()  # class index
    names = result.names                     # class names


    img = image.copy()
    draw = ImageDraw.Draw(img)

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        label = f"{names[int(cls)]} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red")

    st.image(img, caption="Custom bounding boxes", use_column_width=True)



    #st.image(annotated, caption="Detected")
