from ctypes import sizeof
import cv2 as cv
import numpy as np
import pyzed.sl as sl
#import sensorfunctions as sensors


def initialize():
    # net = cv2.dnn.readNetFromDarknet("yolov4-tiny-custom-test.cfg",
    #                             r"yolov4-tiny-custom_last.weights")

    ### define classes
    classes = ['pedestrians', 'speed-30 km', 'speed-50 km', 'speed-60 km', 'stop', 'traffic light - green',
            'traffic light - red', 'vehicle', 'no left turn sign', 'no right turn sign', 'yield', 'tree', 'building',
            'Curved road sign']
    return classes

def get_distance_to_object(depth, x, y):

    err, distance = depth.get_value(x, y)

    return distance

# def display(img, classes, indexes, boxes, class_ids, confidences):
#     cv.imshow("Image", img)
#     font = cv.FONT_HERSHEY_PLAIN  ### font of predicting box text
#     colors = np.random.uniform(0, 255, size=(len(boxes), 3))   ### color of predicting boxes  # generating colors for each object for later plotting
#     print(len(indexes))
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])   ### predicting result class
#             confidence = str(round(confidences[i], 2))   ### predicting result's confidence on the object
#             color = colors[i]  ### yolo box's color for different objects
#             cv.rectangle(img, (x, y), (x + w, y + h), color, 2)  ### yolo predicting box (rectangle)
#             cv.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)  ### text on yolo predicting box (rectangle) which is class+confidence

def inference(img, classes, depth):
    net = cv.dnn.readNetFromDarknet("yolov4-tiny-custom-test.cfg",
                                r"yolov4-tiny-custom_last.weights")

    #img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    height, width, channels = img.shape

    blob = cv.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)  # create 4D blob
    ### input to the network is a so-called blob object. A blob is a 4D numpy array object (images, channels, width, height).
    ### USing blob function of opencv to preprocess image

    net.setInput(blob)   # sets the blob as the input of the network

    output_layers_name = net.getUnconnectedOutLayersNames()  # get all the layer names
    ### use “outputlayers = net.getUnconnectedOutLayersNames()” to simplify
    ### “layer_names = net.getLayerNames(); outputlayers=[layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]”
    ### they are the same

    layerOutputs = net.forward(output_layers_name)

    boxes = []
    confidences = []
    class_ids = []
    distances = []

    for output in layerOutputs:  # loop over each of the layer outputs
        for detection in output:   # loop over each of the object detections
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]  # extract the class id (label) and confidence (as a probability) of  # the current object detection
            if confidence > 0.5: # 0.5   # discard weak predictions by ensuring the detected   # probability is greater than the minimum probability
                center_x = int(detection[0] * width) ### location of the image?
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)   ### size of the image?
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                #if object of interest spotted, determine distance
                if class_id ==0 or class_id == 4 or class_id == 5 or class_id == 6 or class_id == 7:
                    distance_to_object = get_distance_to_object(depth, center_x, center_y) #REPLACE img with depth from ZED
                    #print(distance_to_object)
                    distances.append(distance_to_object)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, .8, .4)

    #display(img, classes, indexes, boxes, class_ids, confidences)
    return class_ids, distances, indexes, boxes, confidences, classes