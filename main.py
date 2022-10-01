import cv2
import torch
from PIL import Image
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.multi_label = True
camera = cv2.VideoCapture(1)
cv2.namedWindow("test")

toothbrush_classes = [79, 67]
drawing = []
while True:
    w, h = 1920//2, 1080//2
    success, image = camera.read()
    if success:
        img = image
        result = model(np.array(img), size=640)
        cv2.imshow('Screen', result.render()[0])

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        results = result.pandas().xyxy[0]
        toothbrush = results[results['class'].isin(toothbrush_classes)]
        xmin, ymin, xmax, ymax = toothbrush.xmin, toothbrush.ymin, toothbrush.xmax, toothbrush.ymax
        drawing_point = (np.mean((xmin, xmax)), ymax)
        drawing.append(drawing_point)
        print(results)
    else:
        print('Waiting')

print(drawing)