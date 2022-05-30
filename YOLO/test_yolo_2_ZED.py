from ctypes import sizeof
import cv2
import numpy as np
import pyzed.sl as sl

### provide the path for testing config file and trained model form colab # load the YOLO network
net = cv2.dnn.readNetFromDarknet("yolov4-tiny-custom-test.cfg",
                                 r"yolov4-tiny-custom_last.weights")

### define classes
classes = ['pedestrians', 'speed-30 km', 'speed-50 km', 'speed-60 km', 'stop', 'traffic light - green',
           'traffic light - red', 'vehicle', 'no left turn sign', 'no right turn sign', 'yield', 'tree', 'building',
           'Curved road sign']

#cap = cv2.VideoCapture(0) ### change camera
def get_distance_to_object(depth_image, x, y):

    err, distance = depth_image.get_value(x, y)

    return distance

def initialize_camera():
    zed = sl.Camera()
    
    init_params = sl.InitParameters()
    init_params.sdk_verbose = True
    init_params.camera_fps=30
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error {}, exiting program".format(err))
        exit()
        
    return zed    

def capture_frame(zed):    
    image = sl.Mat()
    depth = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        
    return [image, depth]


def main():
    zed = initialize_camera()
    while 1:
        
        image, depth = capture_frame(zed)
        img = image.get_data()
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        height, width, channels = img.shape
    
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)  # create 4D blob
        ### input to the network is a so-called blob object. A blob is a 4D numpy array object (images, channels, width, height).
        ### USing blob function of opencv to preprocess image
    
        net.setInput(blob)   # sets the blob as the input of the network
    
        output_layers_name = net.getUnconnectedOutLayersNames()  # get all the layer names
        ### use “outputlayers = net.getUnconnectedOutLayersNames()” to simplify
        ### “layer_names = net.getLayerNames(); outputlayers=[layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]”
        ### they are the same
    
        layerOutputs = net.forward(output_layers_name)
    
        ### Showing information on the screen
        """
        boxes = []
        confidences = []
        class_ids = []
    
        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.7: #0.7  # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
        """
    
        boxes = []
        confidences = []
        class_ids = []

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
                    if class_id == 4 or class_id == 5 or class_id == 6 or class_id == 7:
                        distance_to_car = get_distance_to_object(depth, center_x, center_y)
                        print(distance_to_car)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
        ### idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
        ### indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold,top_k=1)
        ### nms_threshold: the IOU (Intersection over Union between different overlapping predicting boxes) threshold used in non-maximum suppression. Reducing this value will make it easier to remove redundant detections.
        ### How cv2.dnn.NMSBoxes works?  https://stackoverflow.com/questions/66701910/why-nmsboxes-is-not-eleminating-multiple-bounding-boxes
    
        font = cv2.FONT_HERSHEY_PLAIN  ### font of predicting box text
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))   ### color of predicting boxes  # generating colors for each object for later plotting
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])   ### predicting result class
                confidence = str(round(confidences[i], 2))   ### predicting result's confidence on the object
                color = colors[i]  ### yolo box's color for different objects
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  ### yolo predicting box (rectangle)
                cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)  ### text on yolo predicting box (rectangle) which is class+confidence

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):   ###press "q" to quit opencv
            break
    
    cap.release()
    cv2.destroyAllWindows()
    zed.close()
main()