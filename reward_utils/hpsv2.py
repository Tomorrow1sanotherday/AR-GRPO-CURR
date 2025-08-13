import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
from typing import Union
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map

warnings.filterwarnings("ignore", category=UserWarning)

def initialize_model(ckpt_path: str = None, device=None, hps_version: str = "v2.0"):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocess_train, preprocessor = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if ckpt_path is None:
        ckpt_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()
    return model, preprocessor, tokenizer

def score(img_or_path: Union[list, str, Image.Image], prompt: str, model=None, preprocessor=None, tokenizer=None) -> list:

    if model is None:
        model, preprocessor, tokenizer=initialize_model()

    model_pack=(model, preprocessor, tokenizer)
    device = next(model.parameters()).device

    if isinstance(img_or_path, list):
        result = []
        for one_img_or_path in img_or_path:
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                if isinstance(one_img_or_path, str):
                    image = preprocessor(Image.open(one_img_or_path)).unsqueeze(0).to(device=device, non_blocking=True)
                elif isinstance(one_img_or_path, Image.Image):
                    image = preprocessor(one_img_or_path).unsqueeze(0).to(device=device, non_blocking=True)
                else:
                    raise TypeError('The type of parameter img_or_path is illegal.')
                # Process the prompt
                text = tokenizer([prompt]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    outputs = model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(float(hps_score[0]))    
        return result,model_pack
    elif isinstance(img_or_path, str):
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            image = preprocessor(Image.open(img_or_path)).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return float(hps_score[0]),model_pack
    elif isinstance(img_or_path, Image.Image):
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            image = preprocessor(img_or_path).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return float(hps_score[0]),model_pack
    else:
        raise TypeError('The type of parameter img_or_path is illegal.')