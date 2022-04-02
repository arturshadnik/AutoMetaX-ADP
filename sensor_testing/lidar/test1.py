# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:53:55 2022

@author: artur
"""

import numpy as np
import matplotlib.pyplot as plt
from rplidar import RPLidar

def initialize_lidar():
    lidar = RPLidar('COM3', baudrate=115200)
    
    info = lidar.get_info()
    print(info)
    health = lidar.get_health()
    print(health)
    
    return lidar

def collect_data(lidar):
    for i, scan in enumerate(lidar.iter_scans()):
        scan_array = np.asarray(scan, dtype=float)
        quality = (scan_array[:,0])
        theta = scan_array[:,1]
        distance = scan_array[:,2]
        if i%2 == 0:
            plot_data(theta, distance)


def plot_data(theta, distance):
    count = 0
    # Turn on interactive plot options
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

    count += 1

    
    
def main():
    i = 0
    lidar = initialize_lidar()
    collect_data(lidar)
        #plot_data(theta, distance)
        #i +=1
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    
#if __name__ == '__main__':
main()