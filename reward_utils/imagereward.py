from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import ImageReward as RM

class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.model_path = "ImageReward-v1.0"
        self.device = device
        self.dtype = dtype
        self.model = RM.load(self.model_path, device=device).eval().to(dtype=dtype)
        self.model.requires_grad_(False)
        
    @torch.no_grad()
    def __call__(self, prompts, images):
        rewards = []
        for prompt,image in zip(prompts, images):
            _, reward = self.model.inference_rank(prompt, [image])
            rewards.append(reward)
        return rewards

def create_model(device=None):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    return scorer

def imagereward_score(images, prompts, model=None):
    if model is None:
        model=create_model()
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
    prompts = [prompt for prompt in prompts]
    
    scores = model(prompts, images)
    return scores, model