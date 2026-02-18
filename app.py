import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ":0"

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
    image_raw = Image.open(uploaded).convert("RGB")
    

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


    if scores and any(s > 0.1 for s in scores):

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            label = f"{names[int(cls)]} {score:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline="blue", width=6)
            draw.text((x1, y1 - 15), label, fill="blue")


        st.image(img, caption="Waldo Detected", use_column_width=True)

    else:
        
        st.info("No detections above threshold.")
        st.image(image_raw)



    #st.image(annotated, caption="Detected")
