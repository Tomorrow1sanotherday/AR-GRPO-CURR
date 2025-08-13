import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.ImageDraw as PImageDraw
from tqdm import tqdm
import glob
from PIL import Image
import sys
import time
import argparse
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from autoregressive.modeling import ImgGen_HF,ImgGen_HF_CONFIG,ImgGenTextImageProcessor

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}

parser=argparse.ArgumentParser()
parser.add_argument("--exp_name",type=str,required=True)
parser.add_argument("--step",default=300,type=int)
parser.add_argument("--cfg",default=2.0,type=float)
parser.add_argument("--out_rt",default="test_results/test.npz",type=str)
parser.add_argument("--split",default=None,type=str,help="generate a subset according to split, used to do paralled generate.")
args=parser.parse_args()

split=args.split
exp_name=args.exp_name
steps=args.step
out_rt=args.out_rt
cfg=args.cfg
if split is not None:
    assert((len(split)&1)==0),"split should be divded by 2"
    h=len(split)>>1
    split_m=int(split[:h])
    split=int(split[h:])
    print("using split",split,"/",split_m)

# download checkpoint
ckpt_path=f"./checkpoints/{exp_name}/checkpoint-{steps}"

# build vae, var
print("building models...")
model=ImgGen_HF.from_pretrained(ckpt_path,device="cuda",torch_dtype=torch.float16)
model.to(model.model_device)
print(next(model.vq_model.parameters()).device)
print(next(model.gpt_model.parameters()).device)
model.freeze_model()
decoder=ImgGenTextImageProcessor(model.text_encoder,model.vq_model,model.config.codebook_embed_dim)
print(f'prepare finished.')
############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
"""
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
more_smooth = False # True for more smooth output
"""

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

def save_npz(sample_folder: str):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    assert len(pngs) == 50_000, f'{len(pngs)} png files found in {sample_folder}, but expected 50,000'
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (50_000, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{sample_folder}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path

def batched_img(img_batch):
    for i in img_batch:
        img_np=i.permute(1,2,0).mul_(255).cpu().numpy()
        yield Image.fromarray(img_np.astype(np.uint8))

def save_png(img,img_p):
    img_dir=os.path.dirname(img_p)
    os.makedirs(img_dir,exist_ok=True)
    img.save(img_p)

# sample
exp_name=exp_name
exp_name=os.path.splitext(exp_name)[0]
for i in tqdm(range(1000)): # for every class sample 50 images
    if split is not None:
        if i%split_m!=split:
            continue
    B = 50
    has_flg=True
    for jdx in range(B):
        fn=f"{out_rt}/{i:>04}{jdx:>02}.png"
        if not os.path.exists(fn):
            has_flg=False
    if has_flg:
        continue
    class_labels = [i]*B
    label_B: torch.LongTensor = torch.tensor(class_labels, device=model.get_model_device()).unsqueeze(-1)
    with torch.inference_mode(),torch.device(model.model_device):
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            #recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
            x = model.generate(label_B, cfg_scale=cfg, top_p=1.0, top_k=0, return_logits=False).output_ids
            print(x.shape)
            x = decoder.batch_decode(x)
            for jdx,j in enumerate(batched_img(x)):
                save_png(j,f"{out_rt}/{i:>04}{jdx:>02}.png")
if split is None or split is not None and split==0:
    time.sleep(60)
    save_npz(out_rt)
