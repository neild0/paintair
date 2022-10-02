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
        nparr = np.fromstring(request.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img = generator.generate(
            "a high quality sketch of the sun , watercolor , pencil color",
            num_steps=50,
            unconditional_guidance_scale=7.5,
            temperature=1,
            batch_size=1,
            input_image="black.png",
            input_image_strength=0.8
        )
        pil_img = Image.fromarray(img[0])

        return {'pic': json.dumps(np.array(pil_img).tolist())}
        # pil_img.save("output.png")