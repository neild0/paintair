import cv2
import torch
from PIL import Image
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
camera = cv2.VideoCapture(1)
cv2.namedWindow("test")

while True:
    w, h = 1920//2, 1080//2
    success, image = camera.read()
    if success:
        img = image
        cv2.imshow("test", img)
        screen = np.array(img)

        result = model(screen, size=640)
        cv2.imshow('Screen', result.render()[0])

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        print('Waiting')