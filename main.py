import cv2
import torch
from PIL import Image
import numpy as np
import mediapipe as mp

from collections import deque
# import speech_recognition as sr
sr = None

def angle_btw_points(point1, point2, base):
    a = np.array([abs(base[0] - point1[0]), abs(base[1] - point1[1])])
    b = np.array([abs(base[0] - point2[0]), abs(base[1] - point2[1])])

    inner = np.inner(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)

    return deg

def dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2

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


if sr: 
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
    img = image

    img = cv2.flip(img, 1)

    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # checking whether a hand is detected
    h, w, c = img.shape
    if results.multi_hand_landmarks:
        index_tip8, index_mcp5, middle_tip12 = None, None, None
        for handLms in results.multi_hand_landmarks: # working with each hand
            index_tip8 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
            middle_tip12 = int(handLms.landmark[12].x * w), int(handLms.landmark[12].y * h)
            wrist0 = int(handLms.landmark[0].x * w), int(handLms.landmark[0].y * h)
            thumb_tip4 = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)

            print("dist is ", dist(index_tip8, middle_tip12))
            if (angle_btw_points(index_tip8, thumb_tip4, wrist0) <= 30.0
                and dist(index_tip8, middle_tip12) > 15000):
                cx, cy = index_tip8
                cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                pts.appendleft((cx, cy))
            else:
                pts.appendleft(None)
    
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

print(drawing)