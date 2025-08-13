"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could
measure the similarity of cross modalities. Please find more information from
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate
the mean average of cosine similarities.

See --help to see further details.

Code adapted from https://github.com/mseitzer/pytorch-fid and
https://github.com/openai/CLIP.

Copyright 2025 The Chinese University of Hong Kong, Shenzhen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
IMAGE_EXTENSIONS = {
    'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'
}

TEXT_EXTENSIONS = {'txt'}
CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"

def init_clip(device=None):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    print('Loading CLIP model: {}'.format(CLIP_MODEL_PATH))
    model = AutoModel.from_pretrained(CLIP_MODEL_PATH).to(device)
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_PATH)
    return model,processor,tokenizer

@torch.no_grad()
def calculate_one_clip_score(pred, target, pred_type, target_type, model=None):
    if model is None:
        model,_,_=init_clip()
    fake_features = forward_modality(model, pred, pred_type)
    real_features = forward_modality(model, target, target_type)

    # normalize features
    real_features = real_features / real_features.norm(
        dim=1, keepdim=True).to(torch.float32)
    fake_features = fake_features / fake_features.norm(
        dim=1, keepdim=True).to(torch.float32)

    # calculate scores
    score = (fake_features * real_features).sum()
    return score / real_features.shape[0], model

def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    for key in data:
        data[key] = data[key].to(device)
    if flag == 'img':
        features = model.get_image_features(**data)
    elif flag == 'txt':
        features = model.get_text_features(**data)
    else:
        raise TypeError
    return features