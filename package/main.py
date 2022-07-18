"""
    main.py
"""
import datetime
import time
import sys
import os
import platform
import cv2
import numpy as np
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from tracker import *


__author__ = "Sahib Singh Cheema"
nameprogram = "main.py"

# Initalizing some variable
# Initialize the Tracker
tracker = EuclideanDistTracker()

# Initialize the video capture object
cap = cv2.VideoCapture("../video/test.avi")
input_size = 320

# Detection confidence threshold
confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = (0.5)
font_thickness = 2

# Middle cross line position
middle_line_position = 250
up_line_position = middle_line_position - 20
down_line_position = middle_line_position + 20

# Store Coco names in a list
classesFile = "../yolov/classes/coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]
detected_classNames = []
classes_names = ["Car", "Motorbike", "Bus", "Truck"]

# Model Files
modelConfiguration = "../yolov/yolov3-320.cfg"
modelWeigheights = "../yolov/yolov3-320.weights"

# configure the network model

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Define random color for each class

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Setting up the Client

client = InfluxDBClient(url="http://localhost:8086",org="SSC Database",token="RMI4Zw0UB7RtxsCJi1MvtbmabI6Bdbw25JO6d0UcuxqPspIYHVvmHR8uWBKFr96YePY9GVT5CpeYE5WrKD219w==")


def insert_database(type, direction):
    """
        Function for feeding the data to the database
    """
    data = {
        "measurement": "count",
        "tags": {"type": type},
        "fields": {"way": direction}
    }
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write("vehicle", "SSC Database", data)  # change the first two variables for your db


def find_center(x, y, w, h):
    """
     Function for finding the center of the rectangle
    """
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy


# List for storing vehicle count information
temp_up_list = []
temp_down_list = []


def count_vehicle(box_id, img):
    """
        Function for coutning vehicle
    """
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):
        if id not in temp_up_list and id not in temp_down_list:
            temp_up_list.append(id)
            # print("temp_up_list: ", temp_up_list)
            # print("temp_down_list", temp_down_list)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list and id not in temp_up_list:
            temp_down_list.append(id)
            # print("temp_up_list: ", temp_up_list)
            # print("temp_down_list", temp_down_list)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            insert_database(classes_names[index], 2)

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            insert_database(classes_names[index], 1)

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here


def trace_ex(file_out):
    msg = "Esecuzione " + nameprogram + " con " + sys.version + "\n"
    file_out.write(msg)


def postProcess(outputs, img):
    """
        Function for finding the detected object from the network ouput
    """
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w, h = int(det[2]*width), int(det[3]*height)
                    x, y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score 
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    print(detection)
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def realTime():
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(outputNames)
    
        # Find the objects from the network output
        postProcess(outputs, img)

        # Draw the crossing lines

        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord("q"):
            break
    # Release the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


def main():

    i = datetime.datetime.now()
    input_file_log = open("../log/traccia.log", "a")

    plat = sys.platform
    ticks = time.time()

    if "win" in plat:
        computer = os.environ.get('COMPUTERNAME')

    if "linux" in plat:
        computer = os.environ.get('HOSTNAME')

    if "darwin" in plat:
        computer = platform.node()

    msg = computer + ";" + ("%s" % i) + ";" + " Esecuzione " + nameprogram
    msg = msg + " con " + sys.version + "\n"
    msg = str(ticks) + ';' + msg
    trace_ex(input_file_log)
    input_file_log.write(msg)
    # Your code goes here
    print("Running...")
    realTime()


if __name__ == '__main__':
    main()


"""

Query:

from(bucket: "vehicle")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "count")
  |> truncateTimeColumn(unit: 1m)
  |> group(columns: ["_time"], mode:"by")
  |> count(column: "_value")
  |> group(columns: ["host", "_measurement"], mode:"by")
  |> yield(name: "mean")
"""
