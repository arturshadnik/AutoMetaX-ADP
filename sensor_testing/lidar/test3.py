# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 00:05:28 2022

@author: artur
"""
import subprocess
import threading
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

lidar_queue = Queue()
lidar_bins = np.arange(361)
lidar_field = np.zeros((1, 361))
lidar_threshold = 1500
obst_threshold = 4000
pos_threshold = 640
estop = 0
dist_ave = obst_threshold
pos_ave = pos_threshold

def main():
    global pos_threshold

    # Start the lidar interfacing thread
    lidar = LidarThread()
    lidar.start()

    # Start the lidar process thread
    lidar_process = LidarProcessThread()
    lidar_process.start()
    
    lidar_plot = LidarPlotThread()
    lidar_plot.start()
    
class LidarThread(threading.Thread):
    def run(self):
        global lidar_queue
        # Run LiDAR SDK program compiled as executable using com port 3, also create an output pipeline
        lidar_exe = subprocess.Popen(['ultra_simple.exe', 'COM6'], stdout=subprocess.PIPE)
        # For every incoming line in the output pipeline put it into the lidar data queue
        for line in lidar_exe.stdout:
            lidar_queue.put(line)

class LidarProcessThread(threading.Thread):
    def run(self):
        global lidar_field
        global estop
        global dist_ave
        # Initialize moving average queue with size mov_avg_win
        #mov_avg_win = 30
        #dist_q = deque(obst_threshold * np.ones(mov_avg_win), maxlen=mov_avg_win)
        # Initialize raw data storage array
        lidar_field_raw = np.zeros((2, 361))
        # Initialize array that determines scanning cone, currently 20 degrees in front of device
        #lidar_scan = np.concatenate([np.arange(350, 361), np.arange(11)])
        
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
        
            # Set to largest possible value == LiDAR range
            field = np.transpose(lidar_field)
            
            lidar_field_binary = LidarPlotThread.controller(field, lidar_field)
            np.savetxt('D:/ADP/lidar_testing/outputs/data_binary{}.csv'.format(time.strftime("%Y%m%d%H%M%S", time.localtime())), lidar_field_binary, delimiter = ',')

class LidarPlotThread(threading.Thread):
    def run(self):
        count = 0
        # Turn on interactive plot options
        plt.ion()
        while 1:
            # Initialize polar plot

            plt.cla()
            ax = plt.subplot(111, projection='polar')
            # Plot LiDAR data in radians
            ax.scatter(np.deg2rad(lidar_bins), lidar_field[0], s=0.5)
            # Set plot parameters
            ax.set_theta_zero_location("N")
            ax.set_ylim(0, 8000)
            # Plot update frequency
            plt.pause(0.01)
            if (count % 20 == 0):
                np.savetxt('D:/ADP/lidar_testing/outputs/data{}.csv'.format(count), lidar_field, delimiter = ',')
            count += 1
            
    def controller(field, lidar_field):
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