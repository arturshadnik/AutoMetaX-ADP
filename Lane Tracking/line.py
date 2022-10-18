import numpy as np
import cv2 as cv
 
def getROI(image, w, h ):
    height = h
    width = w
    # Defining Triangular ROI: The values will change as per your camera mounts

    triangle = np.array([[(int(width/2), height), (width, height), (width-int(width/4), int(height/1.9))]])
    # creating black image same as that of input image
    black_image = np.zeros_like(image)
    # Put the Triangular shape on top of our Black image to create a mask
    mask = cv.fillPoly(black_image, triangle, color =(255,255,255))
    cv.imshow('mask', mask)
    A= np.array([ [0,0],[0,w],[w,h],[h,0]],dtype="float32")
    B= np.array([ [0,w-int(4*w/5)],[0,w-int(w/5)],[w-1,h-1],[h-1,0]],dtype="float32")
    M=cv.getPerspectiveTransform(A,B)
    mask2=cv.warpPerspective(mask,M,(w,h))
    cv.imshow('mask2', mask2)
    # applying mask on original image
    masked_image = cv.bitwise_and(image, mask2)
    

    return masked_image
#print img.shape # output：(320, 320, 3)
img2 = cv.imread("Test1.jpeg")
size = img2.shape

w = size[1] #with

h = size[0] #height
img2= getROI(img2,w,h)
img = np.zeros((h, w, 3), np.uint8) #grey

ptStart = (160, 320)
ptEnd = (220, 160)
point_color = (0, 255, 0) # BGR
thickness = 1 
lineType = 4
cv.line(img, ptStart, ptEnd, point_color, thickness, lineType)


ptStart = (220, 160)
ptEnd = (300, 0)
point_color = (0, 255, 0) # BGR
thickness = 1
lineType = 8
cv.line(img, ptStart, ptEnd, point_color, thickness, lineType)
contours = np.array([(160, 320),(180, 320), (240, 160)])

cv.fillPoly(img, pts = [contours], color =(0,255,0))
result = cv.addWeighted(img2, 1, img, 0.3, 0)
cv.imshow('ga', result)
cv.namedWindow("img")
cv.imshow('image', img)
cv.waitKey (10000) 
cv.destroyAllWindows()
