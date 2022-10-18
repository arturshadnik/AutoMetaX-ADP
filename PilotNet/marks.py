from collections import deque
import numpy as np
import cv2
import math

vs = cv2.VideoCapture('test_road_2.mp4')
pts = deque(maxlen=64) #buffer size
#size = vs.shape

#w = size[1] #width
#h = size[0] #height
w = vs.get(3)
h = vs.get(4)
print (w,h)

# keep looping
count=0
throttle = 0.1
while True:
    count+=5
    ret,frame = vs.read()

    if frame is None:
        break
    steering=(count)*math.pi/180
    cv2.putText(frame,'Next Step Steering Angle: '+str(steering)[:7]+' deg', (int((5/600)*w), int((80/338)*h)), cv2.FONT_HERSHEY_SIMPLEX, (float((0.5/600)*w)),(255,255,255),2,cv2.LINE_AA) 
    ptStart = (int(w/2), int(h))
    PT1=int(w/2-(h/4)*math.sin(steering*0.0174))
    PT2=int(h)-int((h/4)*math.cos(steering*0.0174))
    ptEnd = (PT1, PT2)
    point_color = (0, 0, 225) # BGR
    thickness = 4 
    lineType = 4
    #print (ptStart,ptEnd)
    cv2.line(frame, ptStart, ptEnd, point_color, thickness, lineType)

    PT3=int(PT1-(h/7)*math.sin(2*steering*0.0174))
    PT4=PT2-int((h/7)*math.cos(2*steering*0.0174))
    ptStart = (PT1, PT2)
    ptEnd = (PT3, PT4)
    point_color = (0, 0, 225) # BGR
    thickness = 4
    lineType = 8
    cv2.line(frame, ptStart, ptEnd, point_color, thickness, lineType) 

    PT5=int(PT3-(h/10)*math.sin(3*steering*0.0174))
    PT6=PT4-int((h/10)*math.cos(3*steering*0.0174))
    ptStart = (PT3, PT4)
    ptEnd = (PT5, PT6)
    point_color = (0, 0, 225) # BGR
    thickness = 4
    lineType = 8
    cv2.line(frame, ptStart, ptEnd, point_color, thickness, lineType) 

    PT7=int(PT5-(h/14)*math.sin(4*steering*0.0174))
    PT8=PT6-int((h/14)*math.cos(4*steering*0.0174))
    ptStart = (PT5, PT6)
    ptEnd = (PT7, PT8)
    point_color = (0, 0, 225) # BGR
    thickness = 4
    lineType = 8
    #cv2.line(frame, ptStart, ptEnd, point_color, thickness, lineType) 

    ptStart = (int(w/2)-20, int(h/2)-20)
    ptEnd = (int(w/2)+20, int(h/2)+20)
    point_color = (0, 225, 0) # BGR
    thickness = 4 
    lineType = 4
    #cv2.line(frame, ptStart, ptEnd, point_color, thickness, lineType)
    ptStart = (int(w/2)-20, int(h/2)+20)
    ptEnd = (int(w/2)+20, int(h/2)-20)
    point_color = (0, 225, 0) # BGR
    thickness = 4 
    lineType = 4
    #cv2.line(frame, ptStart, ptEnd, point_color, thickness, lineType)    
    """
    ptStart = (int(self.center_lane), int(self.height))
    ptEnd = (int(self.center_lane), int(self.height-500))
    point_color = (0, 225, 0) # BGR
    thickness = 4 
    lineType = 4
    cv2.line(image_copy, ptStart, ptEnd, point_color, thickness, lineType)
    """
    point_size = 5
    point_color = (0, 0, 255) # BGR
    thickness = 4 # can be 0 、4、8
    points_list = [(PT1, PT2)]
    for point in points_list:
      cv2.circle(frame, point, point_size, point_color, thickness)
    point_size = 5
    point_color = (0, 0, 255) # BGR
    thickness = 4 # can be 0 、4、8
    points_list = [(PT3, PT4)]
    for point in points_list:
      cv2.circle(frame, point, point_size, point_color, thickness)     
    point_size = 5
    point_color = (0, 0, 255) # BGR
    thickness = 4 # can be 0 、4、8
    points_list = [(PT5, PT6)]
    for point in points_list:
      cv2.circle(frame, point, point_size, point_color, thickness)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()    