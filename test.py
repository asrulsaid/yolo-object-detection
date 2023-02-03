import cv2
from object_detection import *
from utils.plots import plot_one_box
import numpy as np

od = ObjectDetection("yolov7.pt")
cap = cv2.VideoCapture("tamalanrea.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

vid_writer = cv2.VideoWriter(
    'output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while True:
    # Load Image
    _, img = cap.read()

    predict, names, colors = od.detect(img)
    for *xyxy, conf, cls in reversed(predict):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, img, label=label,
                     color=colors[int(cls)], line_thickness=1)
    # Show Image
    # im0 = img.copy()
    # im0 = cv2.resize(img, None, fx=0.5, fy=0.5)
    # cv2.imshow("Img", im0)
    vid_writer.write(img)
    # key = cv2.waitKey(0)
    # if key == 27:
    #     break
cv2.destroyAllWindows()
