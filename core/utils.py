import cv2
import numpy as np
import os
import streamlit as st
import skimage

from deskew import determine_skew
from functools import partial
from skimage.transform import rotate
from skimage.color import rgb2gray
from ultralytics import YOLO

MODEL_DIR = 'models/detection/general'
NUMBER_PLATE_DIR = 'models/detection/num_plate/'
CHARACTER_DIR = 'models/detection/character/'


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


def get_all_detection_models(MODEL_DIR=None):
    return [model_name for model_name in os.listdir(MODEL_DIR) if model_name.endswith('.pt')]


def crop_plate(img_rotated, deskewed=False):
    h, w, _ = img_rotated.shape
    h_crop_percent = 10 if deskewed else 5
    h_crop_value = int(h * h_crop_percent / 100)

    return img_rotated[h_crop_value:-h_crop_value, :]

def deskew_plate(img):
    if isinstance(img, str):
        img = skimage.io.imread(img)
    elif isinstance(img, np.ndarray):
        img = img
    img_gray = rgb2gray(img)
    angle = determine_skew(img_gray)

    if angle is None or angle <=1.5:
        return crop_plate(img.astype(np.uint8))
    img_rotated = rotate(img, angle, resize=False) * 255
    return crop_plate(img_rotated.astype(np.uint8), deskewed=True)


def determine_centroid(character_boxes, character_classes):
    centroid_vals = []
    for char_box in character_boxes:
        x1, y1, x2, y2 = [int(round(val, 0)) for val in char_box.xyxy.squeeze()]
        # centroid calculation of box
        x_mid, y_mid = x1 + int((x1+x2)/2), y1 + int((y1+12)/2)

        label_idx = int(char_box.cls)
        label = character_classes[label_idx]    #type:ignore
        centroid_vals.append({(x_mid, y_mid):label})
    return centroid_vals
