import cv2
import numpy as np
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg")

classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

image_folder = "images"

for image_name in os.listdir(image_folder):

    img = cv2.imread(os.path.join(image_folder,image_name))
    img = cv2.resize(img,(800,600))

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes=[]
    confidences=[]
    class_ids=[]

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    cv2.imshow("Detection",img)

    cv2.waitKey(0)

cv2.destroyAllWindows()