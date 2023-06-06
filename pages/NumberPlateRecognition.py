import cv2
import os
import tempfile
import time
import numpy as np
import streamlit as st

from PIL import Image
from sys import platform

from core.binarize_and_cal_rows import row_count
from core.cluster_points import split_points, sort_points
from core.utils import deskew_plate, determine_centroid
from core.utils import MODEL_DIR, NUMBER_PLATE_DIR, CHARACTER_DIR
from core.utils import load_model, detect, get_extension, get_all_detection_models

st.set_page_config(layout="wide")

INPUT_SOURCE_TYPE = ('File Upload', 'Camera')
TASK_TYPE = ('Image Classification', 'Object Detection')
INPUT_DATA_TYPE = ('Video', 'Image')

INITIAL_DETECTION_THRESHOLD = 0.55
PLATE_DETECTION_THRESHOLD = None
CHARACTER_DETECTION_THRESHOLD = None

number_plate_model = None 
chracter_plate_model = None

# assert that path to model actually points to model.    
assert os.path.exists(MODEL_DIR)

st.write("<B><h2>Inference on Deep Learnig Models</h2></B>", unsafe_allow_html=True)

# Choose task type
task_btn = st.sidebar.radio("Choose Task", TASK_TYPE)

# Image classification isn't configured yet.
if task_btn==TASK_TYPE[0]:
    st.info(f"{task_btn} is yet to configure")
    st.stop()

elif task_btn==TASK_TYPE[1]:
    # Load Plate Detection Model
    number_plate_models = get_all_detection_models(NUMBER_PLATE_DIR)
    model_name = st.sidebar.selectbox(
        "Number Plate Models", [''.join(model.split('.')[:-1]) for model in number_plate_models]
    )
    model_name = os.path.join(NUMBER_PLATE_DIR, model_name + '.pt') # type: ignore
    assert os.path.exists(model_name)

    number_plate_model = load_model(model_name)
    plate_classes = number_plate_model.names
    PLATE_DETECTION_THRESHOLD = st.sidebar.slider("Plate Detection Threshold", min_value=0.00, max_value=1.0, value=INITIAL_DETECTION_THRESHOLD, step=0.01)

    st.sidebar.write("-"*15)

    # Load Character Detection Model
    character_models = get_all_detection_models(CHARACTER_DIR)
    model_name = st.sidebar.selectbox(
        "Character Models", [''.join(model.split('.')[:-1]) for model in character_models]
    )
    model_name = os.path.join(CHARACTER_DIR, model_name + '.pt') # type: ignore
    assert os.path.exists(model_name)

    chracter_plate_model = load_model(model_name)
    character_classes = chracter_plate_model.names
    CHARACTER_DETECTION_THRESHOLD = st.sidebar.slider("Character Detection Threshold", min_value=0.00, max_value=1.0, value=INITIAL_DETECTION_THRESHOLD, step=0.01)
    

if number_plate_model is None or chracter_plate_model is None:
    st.error("Model isn't loaded.")

# Either to use camera or local file as the source
input_btn = st.sidebar.selectbox("Input Type", INPUT_SOURCE_TYPE)


if input_btn==INPUT_SOURCE_TYPE[0]:    
    # When using <File Upload>
    with st.expander("Details"):
        st.info("Input Source is Uploaded file.")
        st.write("You can select image or video file.")
    # Either image or video selection
    input_data_type = st.radio("Choose input medium.", INPUT_DATA_TYPE, key='tab_upload_input_data_type')

    if input_data_type==INPUT_DATA_TYPE[0]:  
        st.write('Not yet Configured')
        st.stop()  
        # video selected
        video_file = st.file_uploader(label="Upload file", type=['.mp4', '.MOV', '.asf', '.avi'])

        if video_file is None:
            st.stop()

        # video_container = st.empty()
        # video_container.video(video_file)

        # splitting into columns to display original and plotted iamge frame
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
            scale_factor = 0.6
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                w, h, _ = frame.shape
                if all([w>1000, h>1800]) or all([h>1000, w>1800]):
                    # real time dection in ui is with scale factor = 0.6 for (1080, 1920) -> (648, 1152)
                    scaled_w, scaled_h = int(round(w*scale_factor, 0)), int(round(h*scale_factor, 0))
                    print((scaled_w, scaled_h))
                    frame = cv2.resize(frame, (scaled_h, scaled_w))
                    print(f"scaling applied.")
                
                pil_img = Image.fromarray(frame)
                img_clone = detect(pil_img, model, DETECTION_THRESHOLD)

                frame_container.image(frame)
                detection_container.image(img_clone)
                
                # paused as hosting machine is linux and requires
                # some time to render in front-end
                if platform == "linux" or platform == "linux2":
                    time.sleep(0.5)
                elif platform=="darwin":
                    time.sleep(0.01)
                elif platform=="win32":
                    time.sleep(0.01)
                
            # Release the VideoCapture object
            cap.release()

            # Update the container with text as completion.
            frame_container.text('Video Finished.')
            detection_container.text('Detection Completed.')

            # Delete the temporary file
            os.remove(video_path)
        
    elif input_data_type==INPUT_DATA_TYPE[1]:  
        # image type selected
        file = st.file_uploader(label="Upload file", type=['.jpg', '.jpeg', '.png'])
        if file is None:
            st.stop()
        filename = file.name
        # img = Image.open(file)
        img = np.array(Image.open(file))

        st.image(img)

        plate_detector_result = number_plate_model.predict(source=img, conf=0.6, verbose=False)    # type:ignore

        plate_boxes = plate_detector_result[0].boxes.cpu().numpy()

        for plate_box in plate_boxes:
            x1, y1, x2, y2 = [int(round(val, 0)) for val in plate_box.xyxy.squeeze()]
            img_cropped = img[y1:y2, x1:x2]

            img_deskewed_cropped = deskew_plate(img_cropped)

            character_detector_result = chracter_plate_model.predict(img_deskewed_cropped, conf=0.6, verbose=False)    # type:ignore

            character_boxes = character_detector_result[0].boxes.cpu().numpy()

            centroid_vals = determine_centroid(character_boxes, character_classes)

            peaks_maxima, peaks_minima, y_axis_vals = row_count(img_deskewed_cropped)   # type:ignore
            h, w, _ = img_deskewed_cropped.shape

            # Sort Centroid values
            rows = split_points(centroid_vals, len(peaks_maxima), peaks_minima, h)
            plate_val = sort_points(rows)
            st.write(plate_val)


