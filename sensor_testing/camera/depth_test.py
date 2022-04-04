# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 00:37:53 2022

@author: artur
"""

import pyzed.sl as sl
import math
import cv2 as cv
import time
from sys import exit


def initialize_camera():
    zed = sl.Camera()
    
    init_params = sl.InitParameters()
    init_params.sdk_verbose = True
    init_params.camera_fps=30
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error {}, exiting program".format(err))
        exit()
        
    return zed    

def capture_image(zed):    
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        
    return [image, depth, point_cloud]

def process_image(image, depth, count):       
    image_cv = image.get_data()
    
    gray = cv.cvtColor(image_cv, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 1)
    edge = cv.Canny(blur, 100, 200)
    cv.imshow('edge image', edge)
    
    x = round(image.get_width() / 2)
    y = round(image.get_height() / 2)
    
    x2 = round(image.get_width() / 3)
    y2 = round(image.get_height() / 4)
    
    #get distance using depth map
    err1, depth_value = depth.get_value(x, y)
    err2, depth_value_2 = depth.get_value(x2, y2)
    
    # print("Distance to Camera at ({0}, {1}): {2} mm".format(x, y, depth_value))
    # print("Distance to Camera at ({0}, {1}): {2} mm".format(x2, y2, depth_value_2))
    
    cv.putText(image_cv,format(round(depth_value,2)), (x+15,y+15),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2, cv.LINE_AA)
    cv.line(image_cv, (x - 20,y) , (x + 20,y), (0,255,0), 2)
    cv.line(image_cv, (x,y - 20) , (x,y + 20), (0,255,0), 2)
    cv.imshow("image", image_cv)
    cv.imwrite("out{}.jpg".format(count), image_cv)

def main():
    zed = initialize_camera()
    count = 0;
    
    bigtic = time.perf_counter()
    
    while True:
        
        tic = time.perf_counter()
        
        [image, depth, point_cloud] = capture_image(zed)
        process_image(image, depth, count)
        count = count + 1
        if cv.waitKey(1) & 0xFF == ord('q'): #exit using "q"
            break
        
        toc = time.perf_counter()
        print("Time to process 1 frame: {}".format(toc-tic))

    bigtoc = time.perf_counter()    
    fps_avg = count/(bigtoc-bigtic)
    print("Average FPS: {}".format(fps_avg))
    
    cv.destroyAllWindows()
    zed.close()
    
if __name__ == '__main__':
    main()