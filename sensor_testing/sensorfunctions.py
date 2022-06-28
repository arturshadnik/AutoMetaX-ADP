"""
Sensor functions library 

Includes initialization/setup functions as well as data collection and preprocessing functions
Designed to work with the ZED Stereo Camera and RoboPeak RPLiDAR A1
Intended for use with the Autonomous Driving Platform

AutoMetaX - 2022

"""

import cv2 as cv
import numpy as np
import pyzed.sl as sl
import math
import subprocess
#import rplidar as RPLidar
from sys import exit

LIDAR_COMM_PORT = 'COM6'
LIDAR_FUNCTION = 'ultra_simple.exe'


######CAMERA FUNCTIONS######

def initialize_camera():
    zed = sl.Camera()
    
    init_params = sl.InitParameters()
    init_params.sdk_verbose = True
    init_params.camera_fps=15
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error {}, exiting program".format(err))
        exit()
        
    return zed   

def capture_frame(zed):    
    image = sl.Mat()
    depth = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        
    return [image, depth]

def get_RGB_image(image):
    image_ocv = image.get_data()
    image_rgb = cv.cvtColor(image_ocv, cv.COLOR_RGBA2RGB)

    return image_rgb

def get_distance_to_object(depth_image, xmin, xmax, ymin, ymax):
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2

    err, distance = depth_image.get_value(x, y)
    
    return distance

def close_camera(zed):
    zed.close()

######LIDAR FUNCTIONS######

def initialize_lidar():
    lidar = RPLidar(LIDAR_COMM_PORT, baudrate=115200)
    
    info = lidar.get_info()
    print(info)
    health = lidar.get_health()
    print(health)
    
    return lidar



def initialize_sensing_system():
    camera = initialize_camera()
    lidar = initialize_lidar()

    return camera, lidar