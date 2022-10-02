from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image
from flask import Flask, flash, request, redirect, url_for
import numpy as np
import cv2
import json

server = Flask(__name__)

generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False,  # You can try True as well (different performance profile)
)

print("server is ready")

@server.route('/sd', methods=['POST'])
def upload_drawing():
    if request.method == 'POST':
        import base64
        print(request)
        print(request.data)
        jpg_as_text = base64.b64decode(request.data)
        img = cv2.imdecode(jpg_as_text, cv2.IMREAD_COLOR)
        cv2.imwrite("black.png", img)
        img = generator.generate(
            "a high quality sketch of the sun , watercolor , pencil color",
            num_steps=50,
            unconditional_guidance_scale=7.5,
            temperature=1,
            batch_size=1,
            input_image="black.png",
            input_image_strength=0.8
        )
        retval, buffer = cv2.imencode('.jpg', img)
        # Convert to base64 encoding and show start of data
        jpg_as_text = base64.b64encode(buffer)
        files = {'img': jpg_as_text}


        return {'img': json.dumps(jpg_as_text)}
        # pil_img.save("output.png")