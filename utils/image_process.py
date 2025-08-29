import tempfile, os, torch
import numpy as np
from torchvision.transforms.functional import to_pil_image as to_pil
import uuid
import torch.distributed as dist


def to_pil_image(x):
    x=x.float()
    return to_pil(x)

def to_temp_path(img, img_name, log_dir: str) -> str:
    # img 可以是 PIL.Image / Tensor / numpy.ndarray / 已经是路径的 str
    if isinstance(img, str):
        return img  
    
    from PIL import Image
    
    # 确保目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    if isinstance(img, torch.Tensor):
        pil = to_pil_image(img)  
        pil = pil.convert("RGB")
    elif isinstance(img, np.ndarray):
        pil = Image.fromarray(img[..., :3]) if img.ndim == 3 else Image.fromarray(img)
        pil = pil.convert("RGB")
    else:  # PIL Image
        pil = img.convert("RGB")
    
    # 生成唯一文件名并保存到指定目录
    rank = dist.get_rank()
    filename = f"{img_name}_{rank}.png"
    file_path = os.path.join(log_dir, filename)
    pil.save(file_path, format="PNG")
    
    return file_path