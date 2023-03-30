import cv2
import numpy as np
import streamlit as st

from copy import deepcopy
from functools import partial
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


def detect(img, model, DETECTION_THRESHOLD):
    result = inference(model, img, DETECTION_THRESHOLD)

    boxes  = result[0].boxes
    boxes = boxes.cpu().numpy()
    classes = result[0].names
    img = result[0].orig_img

    # Wrap with List fuction to get the effect
    list(map(partial(postprocess, classes=classes, img_clone=img), boxes))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return img
