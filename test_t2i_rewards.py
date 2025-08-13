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
import jsonlines
import json
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from autoregressive.modeling import ImgGen_HF,ImgGen_HF_CONFIG,ImgGenTextImageProcessor
from img_gen_grpo_rewards import *

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}

parser=argparse.ArgumentParser()
parser.add_argument("--exp_name",type=str,required=True)
parser.add_argument("--data_json",type=str,default="prompts/evaluation_metadata.jsonl")
parser.add_argument("--dataset",type=str,default="geneval")
parser.add_argument("--eval_root",type=str,default="./benchmark")
parser.add_argument("--sample_num",type=int,default=4)
parser.add_argument("--cfg_scale",type=float,default=7.5)
parser.add_argument("--top_k",type=int,default=2000)
parser.add_argument("--top_p",type=float,default=1.0)
parser.add_argument("--temperature",type=float,default=1.0)
parser.add_argument("--step",default=300,type=int)
parser.add_argument("--out_rt",default="test_results/test.npz",type=str)
parser.add_argument("--delay_load_text_encoder",type=bool,default=True)
parser.add_argument("--split",default=None,type=str,help="generate a subset according to split, used to do paralled generate.")
args=parser.parse_args()

gen_cfg=dict(cfg_scale=7.5, top_k=2000, top_p=1.0, temperature=1.0)
for k in gen_cfg:
    if hasattr(args,k):
        gen_cfg[k]=getattr(args,k)
split=args.split
exp_name=args.exp_name
steps=args.step
out_rt=args.out_rt
if split is not None:
    assert((len(split)&1)==0),"split should be divded by 2"
    h=len(split)>>1
    split_m=int(split[:h])
    split=int(split[h:])
    print("using split",split,"/",split_m)

# download checkpoint
ckpt_path=f"./checkpoints/{exp_name}/checkpoint-{steps}"
print("building models...")
model=ImgGen_HF.from_pretrained(ckpt_path,device="cuda",torch_dtype=torch.bfloat16,delay_load_text_encoder=args.delay_load_text_encoder)
if args.delay_load_text_encoder:
    model.load_text_encoder()
model.to(model.model_device)
print(next(model.vq_model.parameters()).device)
print(next(model.gpt_model.parameters()).device)
model.freeze_model()
processor=ImgGenTextImageProcessor(model.text_encoder,model.vq_model,model.config.codebook_embed_dim)
print(f'prepare finished.')

#######
rewards_dict={
    "clip_score":CLIPTextReward(device=torch.device("cuda:1")),
    "hpsv2":HPSV21TextReward(device=torch.device("cuda:1")),
    "pickscore":PickscoreRewardBatch(device=torch.device("cuda:2")),
    "imagereward":ImageRewardBatch(device=torch.device("cuda:2")),
    "deqa":DeQARewardBatch(device=torch.device("cuda:1")),
    "aesthetic":AestheticRewardBatch(device=torch.device("cuda:1")),
    "unireward":UnifiedRewardBatch(device=torch.device("cuda:1")),
}

rewards_scale_dict={
    "clip_score": lambda x:[xi/5 for xi in x] if type(x) is list else x/5,
    "hpsv2": lambda x:[xi/5 for xi in x] if type(x) is list else x/5,
    "pickscore": lambda x:x,
    "imagereward": lambda x:x,
    "deqa": lambda x:x,
    "aesthetic": lambda x:x,
    "unireward": lambda x:x,
}

def calc_rewards(prompts,imgs):
    res_dict={}
    for k in rewards_dict:
        res_dict[k]=rewards_dict[k](prompts=[{"text":p}for p in prompts],completions=imgs)
    return res_dict
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
        img_np=i.permute(1,2,0).mul_(255).float().cpu().numpy()
        yield Image.fromarray(img_np.astype(np.uint8))

def save_png(img,img_p):
    img_dir=os.path.dirname(img_p)
    os.makedirs(img_dir,exist_ok=True)
    img.save(img_p)

# sample
exp_name=exp_name
exp_name=os.path.splitext(exp_name)[0]

all_data=[]
eval_path=os.path.join(args.eval_root,args.dataset,args.data_json)
if args.dataset=="geneval":
    with jsonlines.open(eval_path) as f:
        all_data=list(f)
elif args.dataset=="drawbench":
    with open(eval_path) as f:
        all_data=json.load(f)

all_res_dict={}

for idx,i in enumerate(tqdm(all_data)): # for every class sample 50 images
    if split is not None:
        if idx%split_m!=split:
            continue
    B = args.sample_num
    if "prompt" in i:
        p=i["prompt"]
    elif "conversations" in i:
        p=i["conversations"][0]["value"]
    else:
        raise ValueError("Not found text")
    print(p)
    prompts = [p]*B
    data_dicts=[{"text":p} for p in prompts]
    input_data=processor(text=data_dicts)
    inputs,attn_mask=input_data["input_ids"],input_data["attention_mask"]
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True),torch.device(model.model_device):    # using bfloat16 can be faster
            x = model.generate(inputs.to("cuda"),attention_mask=attn_mask.to("cuda"), return_logits=False, **gen_cfg).output_ids
            print(x.shape)
            x = processor.batch_decode(x)
        res_dict=calc_rewards(prompts,x)
        for k in res_dict:
            if k not in all_res_dict:
                all_res_dict[k]=[]
            all_res_dict[k].extend(rewards_scale_dict[k](res_dict[k]))
        for jdx,j in enumerate(batched_img(x)):
            save_png(j,f"{out_rt}/{idx:>05}/samples/{jdx:>04}.png")
    with jsonlines.open(f"{out_rt}/{idx:>05}/metadata.jsonl","w") as f:
        f.write(i)

mean_fun=lambda x:sum(x)/len(x)

for k in all_res_dict:
    print(k,mean_fun(all_res_dict[k]))