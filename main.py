import cv2
import torch
from PIL import Image
import numpy as np
import mediapipe as mp
import json
from collections import deque
import speech_recognition as sr
# import replicate
import requests
import json

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

camera = cv2.VideoCapture(1)
cv2.namedWindow("test")

CLEAR = False
END = False

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

toothbrush_classes = [79, 67]
drawing = [None]
black = None
pts = deque()
color_dict = {
              'read': (0, 19, 249), 'red': (0, 19, 249),
              "orange": (32, 171, 255),
              "yellow": (0, 213, 255),
              "green": (6, 214, 54),
              "turquoise": (190, 251, 49),
              "light blue": (255, 214, 152),
              "dark blue": (234, 63, 36),
              "blue": (229, 139, 16),
              "purple": (229, 16, 110),
              "pink": (224, 158, 255),
              "black": (0, 0, 0),
              "white": (255, 255, 255)
              }
# this is called from the background thread
draw_color = (0, 0, 0)
DISPLAY = False

def recognize_whisper(self, audio_data, model="base", show_dict=False, load_options=None, language=None,
                      translate=False, **transcribe_options):
    """
    Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using Whisper.
    The recognition language is determined by ``language``, an uncapitalized full language name like "english" or "chinese". See the full language list at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
    model can be any of tiny, base, small, medium, large, tiny.en, base.en, small.en, medium.en. See https://github.com/openai/whisper for more details.
    If show_dict is true, returns the full dict response from Whisper, including the detected language. Otherwise returns only the transcription.
    You can translate the result to english with Whisper by passing translate=True
    Other values are passed directly to whisper. See https://github.com/openai/whisper/blob/main/whisper/transcribe.py for all options
    """

    assert isinstance(audio_data, sr.AudioData), "Data must be audio data"
    # import whisper
    import tempfile

    if load_options or not hasattr(self, "whisper_model") or self.whisper_model.get(model) is None:
        self.whisper_model = getattr(self, "whisper_model", {})
        self.whisper_model[model] = whisper.load_model(model, **load_options or {})

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(audio_data.get_wav_data())
        f.flush()
        result = self.whisper_model[model].transcribe(
            f.name,
            language=language,
            task="translate" if translate else None,
            **transcribe_options
        )

    if show_dict:
        return result
    else:
        return result["text"]

def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        text = recognizer.recognize_vosk(audio)
        if 'i made' in text:
            global END
            END = True
        if "clear" in text:
            global CLEAR
            CLEAR = True

        # text = recognizer.recognize_whisper(audio)
        for color, value in color_dict.items():
            if f' {color} ' in text or f'{color}' in text:
                global draw_color
                draw_color = value
                break
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

wait_iters = 5
while True:
    w, h = 1920//2, 1080//2
    success, image = camera.read()
    black = np.zeros_like(image) + 255


    if not success:
        continue

    img = image

    img = cv2.flip(img, 1)

    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # checking whether a hand is detected
    h, w, c = img.shape

    if END:
        import base64
        from io import BytesIO

        # make the request
        url = "http://latte.csua.berkeley.edu:5000/sd"
        retval, buffer = cv2.imencode('.jpg', black)
        # Convert to base64 encoding and show start of data
        jpg_as_text = base64.b64encode(buffer)
        prompt = input("What's the prompt?:")
        files = {'img': jpg_as_text, "prompt": prompt}
        response = requests.post(url, json = files).json()
        jpg_as_text = base64.b64decode(eval(response["img"]))
        result = BytesIO(jpg_as_text)
        result = Image.open(result)
        cv2.imwrite("hard.png", np.array(result))

        END = False
        DISPLAY = True
        pts.clear()
        cv2.imshow("Frame", image)
        continue

    if CLEAR:
        pts.clear()
        CLEAR = False
        DISPLAY = False

    # img //= 4
    if not DISPLAY:
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            prev_point, cur_point, color = pts[i-1][:2], pts[i][:2], pts[i][2]
            cv2.line(img, prev_point, cur_point, color, 10)
            cv2.line(black, prev_point, cur_point, color, 10)
        cv2.imwrite('black.png', black)
    else:
        y_offset, x_offset = 0,0
        img[y_offset: y_offset + result.size[0], x_offset: x_offset + result.size[1]] = result
    if results.multi_hand_landmarks:
        index_tip8, index_mcp5, middle_tip12 = None, None, None
        for handLms in results.multi_hand_landmarks: # working with each hand
            index_tip8 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
            middle_tip12 = int(handLms.landmark[12].x * w), int(handLms.landmark[12].y * h)
            wrist0 = int(handLms.landmark[0].x * w), int(handLms.landmark[0].y * h)
            thumb_tip4 = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)
            if (angle_btw_points(index_tip8, thumb_tip4, wrist0) <= 25.0) or thumb_tip4 is None or wrist0 is None or middle_tip12 is None:
                cx, cy = index_tip8
                cv2.circle(img, (cx, cy), 25, draw_color, cv2.FILLED)
                pts.append((cx, cy, draw_color))
                wait_iters = 5
            else:
                cx, cy = index_tip8
                cv2.circle(img, (cx, cy), 15, draw_color, cv2.FILLED)
                wait_iters -= 1
                if not wait_iters:
                    pts.append(None)

    cv2.imshow("Frame", img)


    k = cv2.waitKey(10)
    if k == 27:
        break

# model = replicate.models.get("stability-ai/stable-diffusion")
#
# input = "a ninja"
# for image in model.predict(prompt=f"a high quality sketch of {input} , watercolor , pencil color", init_image=open("black.jpeg", "rb"), width=1024, height=768, prompt_strength=0.7, num_inference_steps=50):
#     print(image)
#
# from stable_diffusion_tf.stable_diffusion import StableDiffusion
# from PIL import Image
#
# generator = StableDiffusion(
#     img_height=512,
#     img_width=512,
#     jit_compile=False,  # You can try True as well (different performance profile)
# )
#
# img = generator.generate(
#     "a high quality sketch of the sun , watercolor , pencil color",
#     num_steps=50,
#     unconditional_guidance_scale=7.5,
#     temperature=1,
#     batch_size=1,
#     input_image="test4.png",
#     input_image_strength=0.8
# )
# pil_img = Image.fromarray(img[0])
