# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 23:48:55 2022

@author: artur
"""

import pyzed.sl as sl
import cv2 as cv
import os

#function to empty a directory
def emptydir(top):
    if(top == '/' or top == "\\"): return
    else:
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))  
                
# Create Camera object
zed = sl.Camera()

# Set config parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
runtime_parameters = sl.RuntimeParameters()
# Open cameras
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)
    

# # Print serial number
# zed_serial = zed.get_camera_information().serial_number
# print(zed_serial) 

# Capture an image
emptydir("D:/ADP/camera_testing/ZED/images")
i = 0
image = sl.Mat()
while i < 50:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT) #Get left image
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) #Get image timestamp
        print("Image resolution: {0} x {1} | Image timestamp: {2}\n".format(image.get_width(),image.get_height(),timestamp.get_milliseconds()))
        i = i + 1
        image_ocv = image.get_data()
        cv.imwrite("D:/ADP/camera_testing/ZED/images/Image{}.jpg".format(i),image_ocv)        

#Close camera
zed.close() 