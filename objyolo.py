import numpy as np
import argparse
import time
import cv2
import os
import pygsheets
from tracker.centroidtracker import CentroidTracker
ct = CentroidTracker()


# gc = pygsheets.authorize(service_file=r'D:\Users\UserPC\Desktop\Synchronizedata-whsh\whchc-sheet-97bc84f48228.json')
# sht = gc.open_by_url('https://docs.google.com/spreadsheets/d/1f1ENRSCCr1QhMuKK9oo-vQQ-YKCrM_2LZ7hGXkxFrs8/')
# wks = sht.worksheet_by_title("Mask")


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
    help="path to input image")
ap.add_argument("-y", "--yolo", required=False,
    help="base path to YOLO directory")
ap.add_argument("-v", "--version", dest='version',
    help="the version of .weights and .cfg file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = r'D:\Users\UserPC\Desktop\Python\final\YoloData\face.names'
weightsPath = r'D:\Users\UserPC\Desktop\Python\final\YoloData\yolov4_final.weights'
configPath = r'D:\Users\UserPC\Desktop\Python\final\YoloData\yolov4.cfg'

LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
# yolo_version = "yolov4" if args["version"] == None else args["version"]

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

cap = cv2.VideoCapture(r'D:\Users\UserPC\Desktop\CENTROID_TRACKER\mt1.mp4')
save = []
def Yrun(net,cap):
    rol = 1
    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (400, 400))
        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        print("[INFO] FPS {:.1f} ".format(1/(end - start)))
        boxes = []
        confidences = []
        classIDs = []
        rects = []
        for output in layerOutputs:
            # rect = []
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    # rect.append(box)
                    # print(box)
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            args["threshold"])
        p = [0,0,0]
        if len(idxs) > 0:
            for i in idxs.flatten():
                print(boxes[i])
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]            
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                if LABELS[classIDs[i]] == 'without_mask' : p[0] += 1 
                elif LABELS[classIDs[i]] == 'with_mask' : p[1] += 1 
                else: p[2] += 1
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)
                print("label: ",text, (x, y), (x + w, y + h))
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            if objectID not in save:
                save.append(objectID)
                cv2.imwrite(f'O-{objectID}.jpg', image)
            cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        try:print('[Without_mask] {:.2f} %'.format((p[0]/sum(p))*100))
        except:pass
        cv2.imshow("Image", image)
        n = time.asctime(time.localtime(time.time()))
        # try:wks.update_value(f'A{rol}', 'danger: {:.2f}% p:{} time: {}'.format((p[0]/sum(p))*100,p,n))
        # except:wks.update_value(f'A{rol}', 'No People {}'.format(n))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rol += 1
        # break
if __name__ == "__main__":
    Yrun(net,cap)
    cap.release()
    cv2.destroyAllWindows()