import cv2
import torch
from PIL import Image
import numpy as np
import mediapipe as mp

<<<<<<< HEAD
=======

import numpy as np
>>>>>>> 68889ebf7132ee0c5a9596c6588a209c2801c800
from collections import deque
import speech_recognition as sr

camera = cv2.VideoCapture(1)
cv2.namedWindow("test")

<<<<<<< HEAD
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

CLEAR = False
END = False

COLOR = BLUE

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

=======
>>>>>>> 68889ebf7132ee0c5a9596c6588a209c2801c800
toothbrush_classes = [79, 67]
drawing = [None]
black = None
pts = deque()
color_dict = {
              'read': (249, 19, 0), 'red': (249, 19, 0),
              "orange": (255, 171, 32),
              "yellow": (255, 213, 0),
              "green": (54, 214, 6),
              "turquoise": (49, 251, 190),
              "light blue": (152, 214, 255),
              "dark blue": (36, 63, 234),
              "blue": (16, 139, 229),
              "purple": (110, 16, 229),
              "pink": (255, 158, 224),
              "black": (0, 0, 0),
              "white": (255, 255, 255)
              }
# this is called from the background thread
draw_color = (0, 0, 0)
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        text = recognizer.recognize_vosk(audio)
        for color, value in color_dict.items():
            if color in text:
                global draw_color
                draw_color = value
                break
        print("Google Speech Recognition thinks you said " + text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

# start listening in the background (note that we don't have to do this inside a `with` statement)
r.listen_in_background(m, callback, phrase_time_limit=2)
# `stop_listening` is now a function that, when called, stops

while True:
    w, h = 1920//2, 1080//2
    success, image = camera.read()

    if not success:
        break
    
    black = np.zeros_like(image) + 255
<<<<<<< HEAD
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
=======
    if success:
        img = image
    
        img = cv2.flip(img, 1)
        kernel = np.ones((5, 5), np.uint8)
        Lower_green = np.array((51,48, 183)) - 20
        Upper_green = np.array((74,65,225)) + 20
        
        mask = cv2.inRange(img, Lower_green, Upper_green)
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
                cv2.circle(img, (int(x), int(y)), int(radius), draw_color, 2)
                cv2.circle(img, center, 5, draw_color, -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
        else:
            pts.appendleft(None)
        
        if END:
            break
>>>>>>> 68889ebf7132ee0c5a9596c6588a209c2801c800

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

print(drawing)