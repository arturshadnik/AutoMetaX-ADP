# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 23:22:41 2022

@author: artur
"""

import time
import numpy as np
from rplidar import RPLidar
import matplotlib.pyplot as plt

class Sensor:
     
    def __init__(self): 
        self.lidar = RPLidar('COM3', baudrate=115200) 
        self.data = [] 
        time.sleep(5) 

    def get_scan(self):
        for i, scan in enumerate(self.lidar.iter_scans()):
            scan_array = np.asarray(scan, dtype=float)
            quality = (scan_array[:,0])
            theta = scan_array[:,1]
            distance = scan_array[:,2]
            if i > 0:
                break
    
        return quality, theta, distance
            
    def plot_scan(self, theta, distance):
        plt.ion()
    
        # Initialize polar plot
        plt.cla()
        ax = plt.subplot(111, projection='polar')
        # Plot LiDAR data in radians
        ax.scatter(np.deg2rad(theta), distance, s=0.5)
        # Set plot parameters
        ax.set_theta_zero_location("N")
        ax.set_ylim(0, 8000)
        # Plot update frequency
        plt.pause(0.01)
                
    def close_sensor(self):
        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.disconnect()
        
def main():      
    sensor1 = Sensor()
    while 1:
        quality, theta, distance = sensor1.get_scan()
        sensor1.plot_scan(theta, distance)
    sensor1.close_sensor()

if __name__ == "__main__":
    main()