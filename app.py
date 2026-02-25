import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ":0"

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw, ImageFont


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
    
    if (len(scores) != 0):
        if(max(scores)>0.1):
            
            for box, score, cls in zip(boxes, scores, classes):
                #only plot the highest score value
                if(score == max(scores)):
                    x1, y1, x2, y2 = box
                    label = f"{names[int(cls)]}" # {score:.2f}"

                    #box width
                    W, H = img.size
                    # Tune 0.004–0.008 depending on how thick you want the line
                    line_w = max(2, int(min(W, H) * 0.01))
                    
                    font_size = max(14, int(min(W, H) * 0.04))
                    font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)

                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=line_w)
                    draw.text((x1, y1 - 20), label, fill="blue", font=font)

            st.info("Waldo was detected!")
            st.image(img,  use_column_width=True)


        else: 
            st.info("Waldo was not found!")
            st.image(image_raw)

    else:
        
        st.info("Waldo was not found!")
        st.image(image_raw)



    #st.image(annotated, caption="Detected")
