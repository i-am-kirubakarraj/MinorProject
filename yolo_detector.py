# yolo_detector.py

import cv2
import numpy as np

class YOLODetector:
    def __init__(self):
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.layer_names = self.net.getLayerNames()
        out_layer_indices = self.net.getUnconnectedOutLayers()
        self.output_layers = [self.layer_names[i - 1] for i in out_layer_indices]

        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect_vehicles(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(obj[0] * frame.shape[1])
                        center_y = int(obj[1] * frame.shape[0])
                        w = int(obj[2] * frame.shape[1])
                        h = int(obj[3] * frame.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return indexes, boxes, confidences, class_ids
