import numpy as np
import math
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

def YUV16toYUV8(luv_Y, luv_U, luv_V):
    """
    YUV uint16 convert to YUV uint8
    """
    # luv_Y = np.squeeze(luv_image[0,:,:,:])
    # luv_U = np.squeeze(luv_image[1,:,:,:])
    # luv_V = np.squeeze(luv_image[2,:,:,:])

    luv_Ynor = (np.round(((luv_Y - np.min(luv_Y)) / (np.max(luv_Y) - np.min(luv_Y))) * 255)).astype(np.uint8)
    luv_Unor = (np.round(((luv_U - np.min(luv_U)) / (np.max(luv_U) - np.min(luv_U))) * 255)).astype(np.uint8)
    luv_Vnor = (np.round(((luv_V - np.min(luv_V)) / (np.max(luv_V) - np.min(luv_V))) * 255)).astype(np.uint8)

    width = luv_Y.shape[0]
    height = luv_Y.shape[1]

    luv_uint8 = np.zeros((width, height, 3), dtype=np.uint8)
    luv_uint8[:,:,0] = luv_Ynor
    luv_uint8[:,:,1] = luv_Unor
    luv_uint8[:,:,2] = luv_Vnor

    return luv_uint8

def YUV2RGB(yuv):
    """
    YUV uint8 convert to RGB uint8
    """
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235]])
    
    rgb = np.dot(yuv,m)
    rgb[:,:,0] -= 179.45477266423404
    rgb[:,:,1] += 135.45870971679688
    rgb[:,:,2] -= 226.8183044444304
    return rgb


def RGB2YUV(rgb):
    """
    RGB uint8 convert back to YUV uint8
    """
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:,:,1:] += 128.0
    return yuv

def YUV8toYUV16(YUV8, luv_Y, luv_U, luv_V):
    """
    YUV uint8 convert to YUV uint16
    """
    YUV8_Y = np.round(YUV8[:,:,0])
    YUV8_U = np.round(YUV8[:,:,1])
    YUV8_V = np.round(YUV8[:,:,2])

    YUV16_Y = np.round(YUV8_Y * (np.max(luv_Y) - np.min(luv_Y)) / 255 + np.min(luv_Y)).astype(np.uint16)
    YUV16_U = np.round(YUV8_U * (np.max(luv_U) - np.min(luv_U)) / 255 + np.min(luv_U)).astype(np.uint16)
    YUV16_V = np.round(YUV8_V * (np.max(luv_V) - np.min(luv_V)) / 255 + np.min(luv_V)).astype(np.uint16) 

    width = luv_Y.shape[0]
    height = luv_Y.shape[1]

    luv_uint16 = np.zeros((width, height, 3), dtype=np.uint16)
    luv_uint16[:,:,0] = YUV16_Y
    luv_uint16[:,:,1] = YUV16_U
    luv_uint16[:,:,2] = YUV16_V

    return luv_uint16

def boxLocal(detections, luv_image):
    """
    Get localization coordinates of box positions.
    """
    numBox = len(detections)
    rowMin = list()
    rowMax = list()
    colMin = list()
    colMax = list()

    for i in range(numBox):
        box = detections[i]
        rowMin.append(np.int16(box.y_min))
        rowMax.append(np.ceil(box.y_max).astype(np.uint16))
        colMin.append(np.int16(box.x_min))
        colMax.append(np.ceil(box.x_max).astype(np.uint16))
    
    for j in range(numBox):
        # blur three channels of the original image
        row_min = rowMin[j]
        row_max = rowMax[j]
        col_min = colMin[j]
        col_max = colMax[j]
        
        luv_image[0,:,row_min:row_max,col_min:col_max] = 0
        luv_image[1,:,row_min:row_max,col_min:col_max] = 0
        luv_image[2,:,row_min:row_max,col_min:col_max] = 0

    return luv_image
