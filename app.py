from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image
from flask import Flask, flash, request, redirect, url_for
import numpy as np
import cv2
import json
from io import BytesIO



server = Flask(__name__)

generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False,  # You can try True as well (different performance profile)
)

print("server is ready")

@server.route('/sd', methods=['POST'])
def upload_drawing():
    # try:
    if request.method == 'POST':
        import base64
        print(request)
        print(request.data)
        j = json.loads(request.data)
        j = j['img']
        jpg_as_text = base64.b64decode(j)
        img = BytesIO(jpg_as_text)
        img = Image.open(img)
        cv2.imwrite("black.png", np.array(img))
        img = generator.generate(
            "a high quality sketch of the sun , watercolor , pencil color",
            num_steps=1,
            unconditional_guidance_scale=7.5,
            temperature=1,
            batch_size=1,
            input_image="black.png",
            input_image_strength=0.8
        )
        # img = Image.fromarray(img[0])
        retval, buffer = cv2.imencode('.jpg', img[0])
        # Convert to base64 encoding and show start of data
        jpg_as_text = str(base64.b64encode(buffer))
        print(type(jpg_as_text),jpg_as_text)
        files = {'img': jpg_as_text}


        return files
    # except Exception as e:
    #     print(e)
    #     # pil_img.save("output.png")