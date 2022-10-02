import cv2
import torch
from PIL import Image
import numpy as np
import mediapipe as mp

from collections import deque

camera = cv2.VideoCapture(0)
cv2.namedWindow("test")

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

CLEAR = False
END = False

COLOR = BLUE

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

toothbrush_classes = [79, 67]
drawing = [None]
pts = deque()
black = None
while True:
    w, h = 1920//2, 1080//2
    success, image = camera.read()

    if not success:
        break
    
    black = np.zeros_like(image) + 255
    img = image

    img = cv2.flip(img, 1)

    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8: # tip of the pointer finger
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                    pts.appendleft((cx, cy))
    
    if END:
        break

    if CLEAR:
        pts.clear()
        CLEAR = False

    # img //= 4
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(img, pts[i - 1], pts[i], COLOR, 2)
        cv2.line(black, pts[i-1], pts[i], COLOR, 2)

    cv2.imshow("Frame", img)
    #cv2.imshow("black", black)
    cv2.imwrite('black.png', black)
    k = cv2.waitKey(10)
    if k == 27:
        break

    else:
        print('Waiting')