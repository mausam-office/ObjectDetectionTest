import cv2
import os
import tempfile
import time
import streamlit as st

from PIL import Image
from sys import platform

from utils import MODEL_DIR
from utils import load_model, detect, get_extension, get_all_detection_models

if platform=="win32":
    import av
    from streamlit_webrtc import webrtc_streamer



INPUT_SOURCE_TYPE = ('File Upload', 'Camera')
TASK_TYPE = ('Image Classification', 'Object Detection')
INPUT_DATA_TYPE = ('Video', 'Image')
INITIAL_DETECTION_THRESHOLD = 0.55
model = None

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
    models = get_all_detection_models()
    # Model name without file extensions
    model_name = st.sidebar.selectbox(
        "Models", [''.join(model.split('.')[:-1]) for model in models]
    )
    model_name = os.path.join(MODEL_DIR, model_name + '.pt') # type: ignore
    assert os.path.exists(model_name)
    # Loads chosen model 
    model = load_model(model_name)
    # Adjust detection threshold dynamically
    DETECTION_THRESHOLD = st.sidebar.slider("Detection Threshold", min_value=0.00, max_value=1.0, value=INITIAL_DETECTION_THRESHOLD, step=0.01)


if platform=="win32":
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:    # type: ignore
        """Function to perform live webcam detection"""
        frame = frame.to_ndarray(format='bgr24')
        frame = detect(frame, model, DETECTION_THRESHOLD)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")    # type: ignore

if model is None:
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
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (864, 576))
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
        img = Image.open(file)

        img_clone = detect(img, model, DETECTION_THRESHOLD)
        st.image(img_clone)
        

elif input_btn==INPUT_SOURCE_TYPE[1]:  
        # Camera selected
        with st.expander("Details"):
            st.info("Input Source is Camera")
            st.write("Please check camera connection if any issue arises.")
        # Choose either video or image
        input_data_type = st.radio("Choose input medium.", INPUT_DATA_TYPE, key='tab_camera_input_data_type')

        if input_data_type == INPUT_DATA_TYPE[0]:   
            # Video Chosen
            if platform=='win32':
                # when code runs in windows machine
                webrtc_streamer(    # type: ignore
                    key='webcam', 
                    video_frame_callback=video_frame_callback,  #type: ignore
                )
            else:
                st.info(f"Source type {input_data_type} is not yet implemented.")
                st.stop()
            # has issue with ICE and TURN servers
            # elif platform=="linux" or platform=="linux2":
            #     webrtc_streamer(
            #         key='webcam', 
            #         video_frame_callback=video_frame_callback,
            #         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            #     )

        elif input_data_type == INPUT_DATA_TYPE[1]:   
            # Image type selected
            img = st.camera_input(label="Capture Image")
            if img is None:
                st.stop()

            filename = img.name
            print(f"filename is {filename} with extension {get_extension(filename)}")
            assert get_extension(filename) in ['jpg', 'jpeg', 'png']
            img = Image.open(img)

            img_clone = detect(img, model, DETECTION_THRESHOLD)
            st.image(img_clone)
        
        

    

