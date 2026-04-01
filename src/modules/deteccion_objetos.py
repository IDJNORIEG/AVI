# Object Detection Code

import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, model_path, config_path, classes_path):
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.classes = []
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect_objects(self, image_path):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detected_objects = []
        for i in indices:
            box = boxes[i]
            (x, y, w, h) = box
            detected_objects.append((self.classes[class_ids[i], x, y, w, h]))

        return detected_objects

# Usage example:
# detection = ObjectDetection('yolov3.weights', 'yolov3.cfg', 'coco.names')
# result = detection.detect_objects('image.jpg')
