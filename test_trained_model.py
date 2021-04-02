from models.yolo import Model
from models.experimental import attempt_load
import cv2
from torchvision import transforms
import numpy as np

weights = 'yolov5s.pt'

model = attempt_load(weights, map_location='cpu')
img = cv2.imread('test.jpg')
img = cv2.resize(img, (640,640), interpolation = cv2.INTER_AREA)

img_tensor = transforms.ToTensor()(img)
print(img_tensor.shape)
img_tensor = img_tensor.unsqueeze(0)
bbox = model(img_tensor)
print(len(bbox))
print(bbox[0].shape)
