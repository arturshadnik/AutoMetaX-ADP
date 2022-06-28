# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 00:05:28 2022

@author: artur
"""
import subprocess
import threading
from queue import Queue
import numpy as np
import time
import sensorfunctions as sensor
import cv2 as cv
import os

lidar_queue = Queue()
lidar_bins = np.arange(361)
lidar_field = np.zeros((1, 361))
lidar_threshold = 1500

#function to empty a directory, only needed during testing to clear out output location, remove in production
def emptydir(top):
    if(top == '/' or top == "\\"): return
    else:
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))  
def main():
    emptydir("D:/ADP/lidar_testing/outputs")
    # Start the lidar interfacing thread
    lidar = LidarThread()
    lidar.start()

    # Start the control thread
    lidar_process = ControlThread()
    lidar_process.start()
    
class LidarThread(threading.Thread):
    def run(self):
        global lidar_queue
        # Run LiDAR SDK program compiled as executable using com port 3, also create an output pipeline
        lidar_exe = subprocess.Popen(['ultra_simple.exe', 'COM6'], stdout=subprocess.PIPE)
        # For every incoming line in the output pipeline put it into the lidar data queue
        for line in lidar_exe.stdout:
            lidar_queue.put(line)
  
class ControlThread(threading.Thread):
    def run(self):
        vid = cv.VideoCapture(0)

        global lidar_field
        #zed = sensor.initialize_camera()
        # Initialize raw data storage array
        lidar_field_raw = np.zeros((2, 361))
        
        #do data processing, call controllers
        while 1:
            # Data is obtained from the queue
            lidar_data = lidar_queue.get()
            
            # Data is split into its constituent parts
            theta = float(lidar_data.split()[0])
            dist = float(lidar_data.split()[1])
            
            # Bin the data according to analog theta values
            bin_num = np.digitize(theta, lidar_bins)
            if bin_num == 361:
                bin_num = 0
                
            # Assign data based on bin number
            lidar_field_raw[0, bin_num] = dist
            lidar_field = np.fliplr(lidar_field_raw)
            field = np.transpose(lidar_field)
            
            
            ret, frame = vid.read()
            #image, depth = sensor.capture_frame(zed)
            print("tired to take an image")
            #if camera working, beging data collection
            if ret == True:
                cv.imshow('image', frame)
                print("took an image")
            if cv.waitKey(1) & 0xFF == ord('q'): #exit using "q"
                vid.release()
                #sensor.close_camera(zed)
                cv.destroyAllWindows()
                break
            #call dummy controller, returns an 2x360 array of [distance, flag]
            lidar_field_binary = ControlThread.controller_dummy(field, lidar_field)
            # cv.imwrite("D:/ADP/lidar_testing/outputs/frame{}.jpg".format(time.strftime("%Y%m%d%H%M%S", time.localtime())), frame)
            
            #save each lidar array, only do this for testing, not in production
            np.savetxt('D:/ADP/lidar_testing/outputs/data_binary{}.csv'.format(time.strftime("%Y%m%d%H%M%S", time.localtime())), lidar_field_binary, delimiter = ',')
    
    #replace this with whatever controller is needed
    def controller_dummy(field, lidar_field):
        i = 0
        lidar_field_binary = lidar_field
        while i < 361:
            if field[i,0] > 0 and field[i,0] < 1500:
                print("detected")
                lidar_field_binary[1, i] = 1
            else:
                print("not detected")
                lidar_field_binary[1, i] = 0
            i += 1

        return lidar_field_binary
        
if __name__ == "__main__":
    main()