import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class ObjectDetection:
    def __init__(self, weights, device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        print("Loading Object Detection")
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Initialize
        set_logging()
        self.device = select_device(device)
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(
            self.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=stride)  # check img_size

        if half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

    def __parse_image(self, img):
        im0 = img.copy()
        img = letterbox(im0, self.imgsz, auto=self.imgsz != 1280)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type != 'cpu' else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return im0, img

    def detect(self, img):
        # Run inference
        t0 = time.time()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            im0, img = self.__parse_image(img)
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres)

        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

        print(f'Done. ({time.time() - t0:.3f}s)')
        return det, self.names, self.colors
