import cv2
import torch
from PIL import Image
import numpy as np


import numpy as np
from collections import deque

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.multi_label = True
camera = cv2.VideoCapture(0)
cv2.namedWindow("test")

toothbrush_classes = [79, 67]
drawing = [None]
pts = deque(maxlen=512*8)

black = None
while True:
    w, h = 1920//2, 1080//2
    success, image = camera.read()
    
    black = np.zeros_like(image) + 255
    if success:
        img = image
    
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kernel = np.ones((5, 5), np.uint8)
        Upper_green = np.array((167,110, 102))
        Lower_green = np.array((115,57,52))
        
        mask = cv2.inRange(hsv, Lower_green, Upper_green)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.bitwise_and(img, img, mask=mask)
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
        else:
            pts.appendleft(None)

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
            cv2.line(black, pts[i-1], pts[i], (0,0,0), 2)
        # except:
        #     # if drawing and drawing[-1] != None:
        #     #     drawing.append(None)
        #     pass

        # print(drawing)
        # img = result.render()[0]
        # for i in range(1, len(drawing)):
        #     if drawing[i-1] is None or drawing[i] is None:
        #         continue
        #     cv2.line(img, drawing[i - 1], drawing[i], (0, 0, 255), 2)

        cv2.imshow("Frame", img)
        #cv2.imshow("black", black)
        cv2.imwrite('black.png', black)
        k = cv2.waitKey(10)
        if k == 27:
            break

    else:
        print('Waiting')

print(drawing)