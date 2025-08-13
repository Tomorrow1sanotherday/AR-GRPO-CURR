import torch
import numpy as np
from PIL import Image

from torchvision import transforms
from .MANIQA.models.maniqa import MANIQA
from .MANIQA.config import Config
from .MANIQA.utils.inference_process import ToTensor, Normalize
from tqdm import tqdm

class Image:
    def __init__(self, image_path, transform, num_crops=20):
        super(Image, self).__init__()
        if type(image_path) is str:
            self.img_name = image_path.split('/')[-1]
            self.img = Image.open(image_path).convert("RGB")
        elif type(image_path) is np.ndarray:
            self.img_name="np image"
            self.img=image_path
        else:
            self.img_name="PIL image"
            self.img=np.array(image_path)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        #print(self.img.shape)
        new_h = 224
        new_w = 224

        self.img_patches = []
        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)
        
        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample

config = Config({
    # image path
    "image_path": "",

    # valid times
    "num_crops": 20,

    # model
    "patch_size": 8,
    "img_size": 224,
    "embed_dim": 768,
    "dim_mlp": 768,
    "num_heads": [4, 4],
    "window_size": 4,
    "depths": [2, 2],
    "num_outputs": 1,
    "num_tab": 2,
    "scale": 0.8,

    # checkpoint path
    "ckpt_path": "./reward_utils/MANIQA/ckpt_koniq10k.pt",
})

def init_model(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net=MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
    net.load_state_dict(torch.load(config.ckpt_path,weights_only=False), strict=False)
    net = net.to(device)
    return net

def calc_one_img(net, img):
    Img = Image(image_path=img,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
        num_crops=config.num_crops)
    avg_score = 0
    net_device=next(net.parameters()).device
    for i in range(config.num_crops):
        with torch.no_grad():
            net.eval()
            patch_sample = Img.get_patch(i)
            patch = patch_sample['d_img_org'].to(net_device)
            patch = patch.unsqueeze(0)
            score = net(patch)
            avg_score += score
    return avg_score / config.num_crops