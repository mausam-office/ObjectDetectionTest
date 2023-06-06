import cv2
import numpy as np
import matplotlib.pyplot as plt

# import skimage
# from skimage.util import img_as_float
from scipy.signal import find_peaks, savgol_filter


def row_count(img):
    col = 0
    row = 1
    if isinstance(img, str):
        img = cv2.imread(img_path)
    elif isinstance(img, np.ndarray):
        img = img
    else:
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, im_gray_th_otsu = cv2.threshold(img_gray, 200, 255, cv2.THRESH_OTSU)
    # im_gray_th_otsu = 255 - im_gray_th_otsu
    # cv2.imshow('img_with_thresh', im_gray_th_otsu)

    y_axis_vals = cv2.reduce(im_gray_th_otsu, row, cv2.REDUCE_AVG)
    y_axis_vals = y_axis_vals.reshape(-1)

    # y_axis_vals_hat = savgol_filter(y_axis_vals, 51, 3)

    len_x = len(y_axis_vals)
    x = np.arange(len_x)

    peaks_maxima, _ = find_peaks(y_axis_vals, distance=len_x//10, height=y_axis_vals.max()//4,  width=len_x//14)
    # print(f"Number of rows found: {len(peaks_maxima)} at pixels {peaks_maxima}")
    

    # Find local maximas
    peaks_minima, _ = find_peaks(-y_axis_vals, distance=len_x//10, height=max(-y_axis_vals)/1.25)#, width=len_x//16)
    # print(f"old minima {peaks_minima}")
    
    # generates valueerror if peaks_maxima is empty
    # peaks_maxima_max = max(peaks_maxima)
    # peaks_maxima_min = min(peaks_maxima)
    # removable_idx = []
    # for idx, val in enumerate(peaks_minima):
    #     if len(peaks_maxima)>1 and not peaks_maxima_min < val < peaks_maxima_max:
    #         removable_idx.append(idx)
    # removable_idx = np.array(removable_idx)
    # # print(f"removable idxes length {len(removable_idx)} {peaks_minima[removable_idx]}")

    # if len(removable_idx)>0:
    #     peaks_minima = np.delete(peaks_minima, removable_idx)
    
    # print(f"after removing outliers {peaks_minima}")


    # # print(y_axis_vals[peaks_minima], np.argmin(y_axis_vals[peaks_minima]))
    # # peaks_maxima = np.array([20, 92, 199])
    # # peaks_minima = np.array([143, 175, 186, 200]) # 52, 60, 

    new_minima = []
    if len(peaks_maxima) > 1:
        for i in range(len(peaks_maxima)-1):
            lower_bound, upper_bound = peaks_maxima[i:i+2]
            # print('lower_bound, upper_bound ', lower_bound, upper_bound)
            
            minima_sliced = peaks_minima[(peaks_minima>lower_bound) & (peaks_minima<upper_bound)]
            try:
                indx = np.argmin(y_axis_vals[minima_sliced])
                new_minima.append(minima_sliced[indx])
            except ValueError as e:
                # print('\n','>>'*5, e, '\n')
                pass
    else:
        new_minima = []
    peaks_minima = np.array(new_minima)

    # print(f"Number of splits found: {len(peaks_minima)} at pixel {peaks_minima}")

    return peaks_maxima, peaks_minima, y_axis_vals


if __name__=="__main__":
    # img_path = './datasets/projects/PlateRecognition/v1-chars/test/images/PlateV1_nbr_plate412_Affine_0_lia1223.jpg'
    # img_path = './datasets/projects/PlateRecognition/v1-chars/train/images/PlateV1_nbr_plate45_Affine_0_VEA6119.jpg'
    # img_path = './datasets/projects/PlateRecognition/v1-chars/train/delete.jpg'
    img_path = "D:/Mausam/YOLOv8/PlateRecognition/recognitions/2023-06-05/3/-Ba7984Pa.jpg"
    peaks_maxima, peaks_minima, y_axis_vals,  = row_count(img_path)

    plt.plot(y_axis_vals,  color='b')
    plt.plot(peaks_maxima, y_axis_vals[peaks_maxima], "x")
    # plt.plot(peaks_minima, y_axis_vals[peaks_minima], "x")
    plt.show()
