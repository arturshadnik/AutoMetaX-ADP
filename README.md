# AutoMetaX-ADP
SFU MSE capstone project Jan 2022-Aug 2022

Supervisor: Dr. Rad

# Team Members
Xilun Zhang
Artur Shadnik
Delin Ma
Zhen Li
Jaehong Kim

# Demonstration Video (Youtube)

1. CARLA Simulation with built-in Stereo Camera
    https://youtu.be/icksM6xUL_o
    
2. CARLA Simulation with HIL LiDAR
    https://youtu.be/mxcCrYY6-Yc
    
3. OpenCV Lane Navigation (Urban road environment) 
    https://www.youtube.com/watch?v=K_LT5URJhMc

4. PilotNet Lane Navigation (Urban road environment)
    https://youtu.be/7JjLhPtkCg4

5. PilotNet Lane Navigation with transfer learning (SFU Surrey Galleria 4)
    https://youtu.be/7xk02w2fQjw

6. Object Classification using ZED Camera (Yolo-v4-tiny)
    https://youtu.be/_xK3NnWSDJg 

# Project Goal
This project aims to improve the functionalities of a level-2 automated vehicle prototype and algorithms. The expected outcome is level-4 automation in ODD. The expected system architecture is shown in the figure below. The main design criteria of this project are environment perception, high-level control (motion planning) and low-level vehicle control (lateral, longitudinal and speed).

<img width="450" alt="image" src="https://user-images.githubusercontent.com/89050720/190730778-5ca351e4-9907-4bd8-982c-67247b646a22.png">

For more details, please refer to our final report or leave an email at xza213@sfu.ca

# Future Plan
1. Sensors
•	The ZED Camera has bad hardware connection, which might lead to danger actions due to lose of camera data during driving.

•	The RPLiDAR can only generate 2D map. To have a better understanding of the environment, a LiDAR that maps 3D environment is required to detect objects lower than the ego vehicle.

•	One extra camera should be added at the back of our ego vehicle to detect the states of approaching vehicle such as velocity and distance.

•	Sensor Fusion algorithm should be added to combine stereo camera and Lidar to reduce variance and noise. 



2. Perception
•   Broaden the YOLO dataset with images of all classes of objects.
•   Broaden dataset to cater different driving environment such as urban road and indoor environment.
•   Implemented Transfer Learning to our own network. Combine pre-trained network such as ImageNet and MobileNet with self-collected dataset.
•   Ensure the perception system can detect incomplete objects for each class including two overlapping objects such as pedestrian holding a stop sign.

