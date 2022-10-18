#import serial
#import multiprocessing as mp
from unittest import result

import numpy as np
import cv2
import time
#from grabscreen import grab_screen
from tensorflow.keras.models import load_model
from collections import deque
import math


#The following function predicted the steering angle using PilotNet

# Make sure the video file is in the same directory as your code
filename = 'orig_lane_detection_1.mp4'
file_size = (1920,1080) # Assumes 1920x1080 mp4
scale_ratio = 1 # Option to scale to fraction of original size.

# initialize the array (used to calculate the average of last three predictions)
# We want to save the output to a video file
output_filename = 'orig_lane_detection_1_lanes.mp4'
output_frames_per_second = 20.0

def pilotnet_prediction(model, frame):
    prediction = model.predict([frame.reshape(-1,187,335,3)])[0] # School
    #prediction = model.predict([frame.reshape(-1, 66, 200, 3)])[0]
    prediction=prediction[0]
    #store last three predicted steering in an array for averaging
    return prediction
 # or class_label == 'pedestrians'


def main():
    '''Load Modified PilotNet to predict steering'''
    #model = load_model('ADP_straight3.h5') # Udacity
    model = load_model('school_123.h5')  # School
    #model = load_model('ADP_straight4.h5') # transfer learninng

    prev_frame_time = 0
    new_frame_time = 0

    # Load a video
    #vs = cv2.VideoCapture("test_road_2.mp4")
    vs = cv2.VideoCapture("school.mp4")
    pts = deque(maxlen=64)  # buffer size
    w = vs.get(3)
    h = vs.get(4)



    # Create a VideoWriter object so we can save the video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_filename,
                             fourcc,
                             output_frames_per_second,
                             file_size)



    while True:

        ''' Pilot Net'''
        #read frame from the camera
        ret,frame0 = vs.read()
        if frame0 is None:
            break

        #resize the frame as per model's input
        frame = cv2.resize(frame0, (672,188))
        #frame = cv2.resize(frame0, (201, 67))
        frame = frame [1:188, 1:336]
        #frame = frame[1:67, 1:201]
        frame_cpy = frame
        #cv2.imshow('camera', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #get predicted steering from PilotNet
        steering = -22.5*pilotnet_prediction(model, frame) # steering*22.5=steering in degree
        #steering = pilotnet_prediction(model, frame)
        #print (steering)

        cv2.putText(frame0, 'Next Step Steering Angle: ' + str(-steering)[:7] + ' deg',
                    (int((5 / 600) * w), int((80 / 338) * h)), cv2.FONT_HERSHEY_SIMPLEX, (float((0.5 / 600) * w)),
                    (255, 255, 255), 2, cv2.LINE_AA)
        ptStart = (int(w / 2), int(h))
        PT1 = int(w / 2 - (h / 4) * math.sin(steering * 0.0174))
        PT2 = int(h) - int((h / 4) * math.cos(steering * 0.0174))
        ptEnd = (PT1, PT2)
        point_color = (0, 0, 225)  # BGR
        thickness = 4
        lineType = 4
        # print (ptStart,ptEnd)
        cv2.line(frame0, ptStart, ptEnd, point_color, thickness, lineType)

        PT3 = int(PT1 - (h / 7) * math.sin(2 * steering * 0.0174))
        PT4 = PT2 - int((h / 7) * math.cos(2 * steering * 0.0174))
        ptStart = (PT1, PT2)
        ptEnd = (PT3, PT4)
        point_color = (0, 0, 225)  # BGR
        thickness = 4
        lineType = 8
        cv2.line(frame0, ptStart, ptEnd, point_color, thickness, lineType)

        PT5 = int(PT3 - (h / 10) * math.sin(3 * steering * 0.0174))
        PT6 = PT4 - int((h / 10) * math.cos(3 * steering * 0.0174))
        ptStart = (PT3, PT4)
        ptEnd = (PT5, PT6)
        point_color = (0, 0, 225)  # BGR
        thickness = 4
        lineType = 8
        cv2.line(frame0, ptStart, ptEnd, point_color, thickness, lineType)

        PT7 = int(PT5 - (h / 14) * math.sin(4 * steering * 0.0174))
        PT8 = PT6 - int((h / 14) * math.cos(4 * steering * 0.0174))
        ptStart = (PT5, PT6)
        ptEnd = (PT7, PT8)
        point_color = (0, 0, 225)  # BGR
        thickness = 4
        lineType = 8
        cv2.line(frame0, ptStart, ptEnd, point_color, thickness, lineType)

        ptStart = (int(w / 2) - 20, int(h / 2) - 20)
        ptEnd = (int(w / 2) + 20, int(h / 2) + 20)
        point_color = (0, 225, 0)  # BGR
        thickness = 4
        lineType = 4
        cv2.line(frame0, ptStart, ptEnd, point_color, thickness, lineType)
        ptStart = (int(w / 2) - 20, int(h / 2) + 20)
        ptEnd = (int(w / 2) + 20, int(h / 2) - 20)
        point_color = (0, 225, 0)  # BGR
        thickness = 4
        lineType = 4
        cv2.line(frame0, ptStart, ptEnd, point_color, thickness, lineType)

        point_size = 5
        point_color = (0, 0, 255)  # BGR
        thickness = 4  # 0 、4、8
        points_list = [(PT1, PT2)]
        for point in points_list:
            cv2.circle(frame0, point, point_size, point_color, thickness)
        point_size = 5
        point_color = (0, 0, 255)  # BGR
        thickness = 4  # 0 、4、8
        points_list = [(PT3, PT4)]
        for point in points_list:
            cv2.circle(frame0, point, point_size, point_color, thickness)
        point_size = 5
        point_color = (0, 0, 255)  # BGR
        thickness = 4  # 0 、4、8
        points_list = [(PT5, PT6)]
        for point in points_list:
            cv2.circle(frame0, point, point_size, point_color, thickness)


        # Write the frame to the output video file
        result.write(frame0)


        cv2.imshow('camera', frame0)

        #convert steering and throttle to 4 digit numbers
        predicted_steering = str(steering).zfill(4)

        #calculate the FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        print('FPS: ',format(fps,'.2f'), predicted_steering)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()


    # Release the video recording
    result.release()
    cv2.destroyAllWindows()

main()

