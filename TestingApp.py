import os
import tempfile
import cv2
import numpy as np
import streamlit as st

from copy import deepcopy
from functools import partial
from PIL import Image
from ultralytics import YOLO


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def inference(model, img, DETECTION_THRESHOLD):
    return model.predict(source=img, conf=DETECTION_THRESHOLD)


def postprocess(_arr, classes, img_clone):
   pts = _arr.xyxy.squeeze()
   x1, y1, x2, y2 = int(round(pts[0], 0)), int(round(pts[1], 0)), int(round(pts[2], 0)), int(round(pts[3], 0))
   score = round(float(_arr.conf) * 100, 0)
   label_idx = int(_arr.cls)
   box_txt = f"{classes[label_idx]}: {score}%"
   cv2.rectangle(img_clone, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=1)
   cv2.putText(img_clone, box_txt, org=(x1, y2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)\


def get_extension(filename):
    return filename.split('.')[-1]


def detect(img):
    result = inference(model, img, DETECTION_THRESHOLD)

    boxes  = result[0].boxes
    boxes_ = boxes.cpu().numpy()
    classes = result[0].names
    img = result[0].orig_img

    img_clone = deepcopy(img)

    # Wrap with List fuction to get the effect
    list(map(partial(postprocess, classes=classes, img_clone=img_clone), boxes_))
    img_clone = cv2.cvtColor(np.array(img_clone), cv2.COLOR_BGR2RGB)
    # st.image(img_clone)
    return img_clone


INPUT_SOURCE_TYPE = ('File Upload', 'Camera')
TASK_TYPE = ('Image Classification', 'Object Detection')
INPUT_DATA_TYPE = ('Video', 'Image')
DETECTION_THRESHOLD = 0.55

model = load_model('models/best.pt')

task_btn = st.sidebar.radio("Choose Task", TASK_TYPE)

if task_btn==TASK_TYPE[0]:
    st.info(f"{task_btn} is yet to configure")
    st.stop()

elif task_btn==TASK_TYPE[1]:
    DETECTION_THRESHOLD = st.sidebar.slider("Detection Threshold", 0.00, 1.0, step=0.01)


input_btn = st.sidebar.selectbox("Input Type", INPUT_SOURCE_TYPE)

if input_btn==INPUT_SOURCE_TYPE[0]:    # File Upload
    with st.expander("Details"):
        st.info("Input Source is Uploaded file.")
        st.write("You can select image or video file.")
    
    

    input_data_type = st.radio("Choose input medium.", INPUT_DATA_TYPE, key='tab_upload_input_data_type')

    if input_data_type==INPUT_DATA_TYPE[0]:    # video
        video_file = st.file_uploader(label="Upload file", type=['.mp4', '.MOV'])
        if video_file is None:
            st.stop()

        # video_container = st.empty()
        # video_container.video(video_file)

        org_frame, plotted_frame = st.columns(2)

        #  for opencv
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

            cap = cv2.VideoCapture(video_path)
            with org_frame:
                frame_container = st.empty()
            with plotted_frame:
                detection_container = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                img_clone = detect(pil_img)

                frame_container.image(frame)
                detection_container.image(img_clone)
                
            # Release the VideoCapture object
            cap.release()

            # Update the container with text as completion.
            frame_container.text('Video Finished.')
            detection_container.text('Detection Completed.')

            # Delete the temporary file
            os.remove(video_path)
        
    elif input_data_type==INPUT_DATA_TYPE[1]:  # image
        file = st.file_uploader(label="Upload file", type=['.jpg', '.jpeg'])
        if file is None:
            st.stop()
        filename = file.name
        img = Image.open(file)

        img_clone = detect(img)
        st.image(img_clone)
        

elif input_btn==INPUT_SOURCE_TYPE[1]:  # Camera
        with st.expander("Details"):
            st.info("Input Source is Camera")
            st.write("Please check camera connection if any issue arises.")
        
        input_data_type = st.radio("Choose input medium.", INPUT_DATA_TYPE, key='tab_camera_input_data_type')

        if input_data_type == INPUT_DATA_TYPE[0]:
            st.info(f"Source type {input_data_type} is not yet implemented.")
            st.stop()

        img = st.camera_input(label="Capture Image")
        if img is None:
            st.stop()

        filename = img.name
        print(f"filename is {filename} with extension {get_extension(filename)}")
        assert get_extension(filename) in ['jpg', 'jpeg', 'png']
        img = Image.open(img)

        img_clone = detect(img)
        st.image(img_clone)
        
        

    

