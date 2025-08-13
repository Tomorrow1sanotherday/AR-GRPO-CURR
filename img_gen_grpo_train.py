import os
import re
from datetime import datetime
from dataclasses import dataclass, field, asdict, make_dataclass, asdict
from typing import Optional

from PIL import Image
import torch
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
import inspect
from img_gen_grpo_trainer import ImgGenGRPOTrainer
from img_gen_grpo_rewards import *
from lazy_dataset import build_dataset
from autoregressive.modeling import ImgGen_HF,ImgGen_HF_CONFIG,ImgGenTextImageProcessor
# for manunal control or tensor board
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

import torch
from typing import Tuple
from copy import deepcopy

reward_funcs_registry = {
    "clip_distance": CLIPDistanceReward,
    "clip_quantize": CLIPDistanceReward,
    "clip_text": CLIPTextReward,
    "clip_text_quantize": CLIPTextReward,
    "mse": MSEReward,
    "maniqa": MANIQAReward,
    "hpsv2": HPSV21DistanceReward,
    "hpsv2_quantize": HPSV21DistanceReward,
    "hpsv2_text": HPSV21TextReward,
    "hpsv2_text_quantize": HPSV21TextReward,
    "qwenvl_3b": QwenLabelReward,
    "qwenvl_3b_text": QwenTextReward,
    "fid": FIDReward,
    "qwenvl_fake": QwenFakeDiscrimReward,
    "qwenvl_weird": QwenWeirdDiscrimReward,
    "geneval": GenEvalTextRewardBatch,
}

reward_funcs_params={
    "clip_distance": lambda func_dict,device: {"device":device},
    "clip_quantize": lambda func_dict,device: {"quantize":True,"clone_reward":func_dict.get("clip_distance",None),"device":device},
    "clip_text": lambda func_dict,device: {"device":device},
    "clip_text_quantize": lambda func_dict,device: {"quantize":True,"clone_reward":func_dict.get("clip_text",None),"device":device},
    "mse": lambda func_dict,device: {"device":device},
    "maniqa": lambda func_dict,device: {"device":device},
    "hpsv2": lambda func_dict,device: {"device":device},
    "hpsv2_quantize": lambda func_dict,device: {"quantize":True,"clone_reward":func_dict.get("hpsv2",None),"device":device},
    "hpsv2_text": lambda func_dict,device: {"device":device},
    "hpsv2_text_quantize": lambda func_dict,device: {"quantize":True,"clone_reward":func_dict.get("hpsv2_text",None),"device":device},
    "fid": lambda func_dict,device: {"device":device},
    "qwenvl_3b": lambda func_dict,device: {"mode":"transformers","device":device},
    "qwenvl_3b_text": lambda func_dict,device: {"mode":"transformers","device":device},
    "qwenvl_fake": lambda func_dict,device: {"mode":"transformers","clone_reward":func_dict.get("qwenvl_3b_text",func_dict.get("qwenvl_3b",None)),"device":device},
    "qwenvl_weird": lambda func_dict,device: {"mode":"transformers","clone_reward":func_dict.get("qwenvl_fake",None),"device":device},
    "geneval": lambda func_dict,device: {},
}

# ----------------------- Main Script -----------------------
@dataclass
class ImgGenGRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: list(reward_funcs_registry.keys()),
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    exit_step: Optional[int] = field(
        default=225,
        metadata={"help": "Exit when exit_step is reached"},
    )
    vq_ckpt: Optional[str] = field(
        default='./pretrained_models/vq_ds16_c2i.pt',
        metadata={"help": "ckpt of vq model"},
    )
    gpt_ckpt: Optional[str] = field(
        default='./pretrained_models/c2i_B_256.pt',
        metadata={"help": "ckpt of language model"},
    )
    gen_cfg_cfg_scale: Optional[float] = field(
        default=1.0,
        metadata={"help": "generation cfg"},
    )
    force_model_bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "force convert model to bfloat16, llamagen use a confused dtype in different ckpts"},
    )
    use_geneval_train: Optional[bool] = field(
        default=False,
        metadata={"help": "use geneval format json as train data"},
    )

ImgGenModelConfig=make_dataclass("ImgGenModelConfig",
    [(k,type(v),field(default=v))for k,v in ImgGen_HF_CONFIG.self_model_config_attrs.items()]
    ,bases=(ModelConfig,))


class ImgGenGRPOTrainingArguments(GRPOConfig):
    pass

def args_calc(script_args, training_args, model_args): # preproccess argments to calculate some auto args after parsing
    script_args.dataset_type = model_args.gpt_type
    script_args.image_size=model_args.image_size

def build_model(script_args, training_args, model_args):
    model=ImgGen_HF(ImgGen_HF_CONFIG.from_dict(asdict(model_args)))
    model.load_vq(script_args.vq_ckpt)
    model.load_model(script_args.gpt_ckpt)
    model.freeze_model(excluded_list=["gpt_model"])
    if script_args.force_model_bf16:
        model.to(dtype=torch.bfloat16)
    return model

def main(script_args, training_args, model_args):
    print("building models")
    model=build_model(script_args, training_args, model_args)
    if training_args.beta > 0.0:
        with torch.no_grad():
            model.ref_model=build_model(script_args, training_args, model_args)
    model_device=next(model.parameters()).device
    print(model_device)
    print(f"Total parameters: ~{model.total_params/1e6:.2f} MB)")
    print(f"Trainable parameters: ~{model.trainable_params/1e6:.2f} MB)")
    #model_device=torch.device(torch.distributed.get_rank())
    #print(model_device)

    print("building rewards")
    func_dict={}
    for func in script_args.reward_funcs:
        if inspect.isclass(reward_funcs_registry[func]):
            func_dict[func]=reward_funcs_registry[func](**reward_funcs_params[func](func_dict,model_device))  
        else:
            func_dict[func]=reward_funcs_registry[func]
    reward_funcs = [func_dict[i] for i in func_dict]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    print("building datasets")
    dataset = build_dataset(script_args.dataset_name, script_args)

    # processing gen config
    script_args_dict=asdict(script_args)
    generation_cfg={}
    for i in script_args_dict:
        if i.startswith("gen_cfg_"):
            generation_cfg[i.replace("gen_cfg_","",1)]=script_args_dict[i]
    #print(generation_cfg)
    training_args.generation_kwargs=generation_cfg
    #print(training_args.top_k)
    trainer_cls = ImgGenGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=ImgGenTextImageProcessor(model.text_encoder,model.vq_model,model.config.codebook_embed_dim),
        exit_step=script_args.exit_step,
    )
    trainer.remove_callback(TensorBoardCallback)
    tf_swr=SummaryWriter(log_dir="./runs/"+training_args.run_name)
    trainer.tf_callback=TensorBoardCallback(tf_swr)
    trainer.add_callback(trainer.tf_callback)
    trainer.tf_swr=tf_swr

    # Train the model
    #with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
    trainer.train()

    if hasattr(model,"ref_model"):
        del model.ref_model
    # Save and push to hub
    trainer.save_model(training_args.output_dir)

    torch.distributed.barrier()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = TrlParser((ImgGenGRPOScriptArguments, ImgGenGRPOTrainingArguments, ImgGenModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    args_calc(script_args, training_args, model_args)
    #print(training_args)
    #input()
    main(script_args, training_args, model_args)
