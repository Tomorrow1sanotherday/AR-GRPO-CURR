import os, torch
import numpy as np
from torchvision.transforms.functional import to_pil_image as to_pil
import torch.distributed as dist
import datetime
import uuid
import glob
import re

def extract_timestamp_from_filename(filepath):
    """从文件名中提取时间戳进行排序"""
    filename = os.path.basename(filepath)
    try:
        # 使用正则表达式匹配时间戳模式：{description}_{rank}_{YYYYMMDD}_{HHMMSS}_{microseconds}_{uid}.png
        # 匹配模式：任意内容_数字_8位日期_6位时间_数字_6位hex.png
        pattern = r'.*_(\d+)_(\d{8})_(\d{6})_(\d+)_([a-f0-9]+)\.png$'
        match = re.match(pattern, filename)
        
        if match:
            rank, date_str, time_str, microsec_str, uid = match.groups()
            
            # 构建时间戳
            timestamp_str = f"{date_str}_{time_str}"
            dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            timestamp = dt.timestamp()
            
            # 添加微秒
            microsec = int(microsec_str[:6].ljust(6, '0'))
            timestamp += microsec / 1_000_000
            
            return timestamp
            
    except Exception as e:
        print(f"Warning: Could not parse timestamp from {filename}, using file mtime: {e}")
    
    # 降级到修改时间
    return os.path.getmtime(filepath)

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
    
    # Generate a timestamp-based unique filename
    rank = dist.get_rank()
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")  # 添加微秒
    uid = uuid.uuid4().hex[:6]  # 随机 6 位
    
    # 清理img_name中的特殊字符，避免文件名问题
    safe_img_name = re.sub(r'[^\w\s-]', '', img_name).strip()
    safe_img_name = re.sub(r'[-\s]+', '-', safe_img_name)
    
    filename = f"{safe_img_name}_{rank}_{timestamp}_{uid}.png"
    file_path = os.path.join(log_dir, filename)
    pil.save(file_path, format="PNG")

    # 检查文件数，超过则删掉最旧的
    try:
        files = glob.glob(os.path.join(log_dir, "*.png"))
        if len(files) > 1280:
            # 按时间戳排序，删除最旧的文件
            files_with_timestamps = [(f, extract_timestamp_from_filename(f)) for f in files]
            files_with_timestamps.sort(key=lambda x: x[1])
            
            files_to_remove = files_with_timestamps[:len(files) - 128]
            for file_to_remove, _ in files_to_remove:
                try:
                    os.remove(file_to_remove)
                    print(f"Removed old file: {os.path.basename(file_to_remove)}")
                except OSError as e:
                    print(f"Warning: Could not remove {file_to_remove}: {e}")
    except Exception as e:
        print(f"Error during file cleanup: {e}")
    
    return file_path
