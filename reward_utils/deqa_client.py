import requests
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
import pickle
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

def create_session():
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    return sess

def deqa_score(images, sess=None, only_strict=False, batch_size = 64, url = "http://127.0.0.1:18086"):
    if sess is None:
        sess=create_session()
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
    all_scores = []
    for image_batch in images_batched:
        jpeg_images = []

        # Compress the images using JPEG
        for image in image_batch:
            img = Image.fromarray(image)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            jpeg_images.append(buffer.getvalue())

        # format for LLaVA server
        data = {
            "images": jpeg_images,
        }
        data_bytes = pickle.dumps(data)

        # send a request to the llava server
        response = sess.post(url, data=data_bytes, timeout=120)
        response_data = pickle.loads(response.content)

        all_scores += response_data["outputs"]

    return all_scores, sess
