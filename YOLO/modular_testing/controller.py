import numpy as np
import cv2 as cv
import yolo
import sensorfunctions as sensors

#use this function if testing with a regular camera, otherwise use initialization function in sensorfunctions library
def initialize_camera():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

#use this function if testing with a regular camera, otherwise use collection function in sensorfunctions library

def collect_image(cap):
    ret, frame = cap.read()
    #if camera working, beging data collection
    if ret == True:
        return frame

def display(img, classes, indexes, boxes, class_ids, confidences, depths):
    font = cv.FONT_HERSHEY_PLAIN  ### font of predicting box text
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))   ### color of predicting boxes  # generating colors for each object for later plotting
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])   ### predicting result class
            label2 = str(depths[i])
            confidence = str(round(confidences[i], 2))   ### predicting result's confidence on the object
            color = colors[i]  ### yolo box's color for different objects
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)  ### yolo predicting box (rectangle)
            cv.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)  ### text on yolo predicting box (rectangle) which is class+confidence
            cv.putText(img, label2, (x + 200, y - 100), font, 2, color, 2)  ### text on yolo predicting box (rectangle) which is class+confidence

    cv.imshow("Image", img)

def main():
    #intialize camera
    cap = sensors.initialize_camera()
    classes = yolo.initialize()

    while True:
        image_zed, depth = sensors.capture_frame(cap)
        image = sensors.get_RGB_image(image_zed)

        objects, depths, indexes, boxes, confidences, classes = yolo.inference(image, classes, depth)
        print(objects)
        print(depths)
        display(image, classes, indexes, boxes, objects, confidences, depths)
        if cv.waitKey(1) == ord('q'):   ###press "q" to quit opencv
            break

    sensors.close_camera(cap)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()