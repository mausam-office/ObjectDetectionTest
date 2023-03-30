import os
import tempfile
import cv2
import streamlit as st

from PIL import Image
from utils import MODEL_DIR
from utils import load_model, detect, get_extension, get_all_detection_models


INPUT_SOURCE_TYPE = ('File Upload', 'Camera')
TASK_TYPE = ('Image Classification', 'Object Detection')
INPUT_DATA_TYPE = ('Video', 'Image')
DETECTION_THRESHOLD = 0.55
model = None
assert os.path.exists(MODEL_DIR)

st.write("<B><h2>Inference on Deep Learnig Models</h2></B>", unsafe_allow_html=True)

task_btn = st.sidebar.radio("Choose Task", TASK_TYPE)

if task_btn==TASK_TYPE[0]:
    st.info(f"{task_btn} is yet to configure")
    st.stop()

elif task_btn==TASK_TYPE[1]:
    models = get_all_detection_models()
    model_name = st.sidebar.selectbox(
        "Models", [''.join(model.split('.')[:-1]) for model in models]
    )
    model_name = os.path.join(MODEL_DIR, model_name + '.pt')
    assert os.path.exists(model_name)
    model = load_model(model_name)
    
    DETECTION_THRESHOLD = st.sidebar.slider("Detection Threshold", 0.00, 1.0, step=0.01)

if model is None:
    st.error("Model is not loaded.")

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
                img_clone = detect(pil_img, model, DETECTION_THRESHOLD)

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

        img_clone = detect(img, model, DETECTION_THRESHOLD)
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

        img_clone = detect(img, model, DETECTION_THRESHOLD)
        st.image(img_clone)
        
        

    
