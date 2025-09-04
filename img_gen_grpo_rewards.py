import sys
from abc import ABC,abstractmethod
from datetime import datetime
from torchvision.transforms.functional import to_pil_image as to_pil
import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace
from typing import List, Dict
import json
import t2v_metrics
from utils.image_process import to_temp_path
import os

def to_pil_image(x):
    x=x.float()
    return to_pil(x)

def discrete_reward_map(value,value_list,reward_list): # assert value list are sorted from low to high
    len_v,len_r=len(value_list),len(reward_list)
    assert value_list[0]<value_list[1]
    assert len_r>=len_v
    if value<value_list[0]:
        return reward_list[0]
    for idx in range(len_v-1):
        if value_list[idx]<=value<value_list[idx+1]:
            return reward_list[idx+1]
    return reward_list[len_v]

def reward_quantize(value,min_v,max_v,min_q=0,max_q=5,num_bins=10): # map value from [min_v,max_v) to [min_q,max_q) and quantize to times of num_bins
    nv=(value-min_v)/(max_v-min_v)
    qv=nv*(max_q-min_q)+min_q
    bin_width=(max_q-min_q)/num_bins
    qv_num=int(qv/bin_width+0.5)
    return bin_width*qv_num

class GRPORewardFunc(ABC):
    def __init__(self,device=None,**kwargs):
        self.__name__=self.__class__.__name__
        self.model_device=device
    @abstractmethod
    def _reward_func(self, cont, sol, **kwargs):
        pass

    def move_to_device(self, device):
        if hasattr(self,"reward_model"):
            self.reward_model.to(device)
            self.model_device=device # record device change
        else:
            print(f"{self.__name__} has no reward model, skip moving device")
    
    def __call__(self, prompts=None, completions=None, **kwargs):
        solutions=[p["image"] for p in prompts]
        if type(completions) is list:
            contents=[[] for i in completions[0]]
            for i in completions:
                for cidx,c in enumerate(i):
                    contents[cidx].append(c)
        else:
            contents = [completion for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol in zip(contents, solutions):
            reward = 0.0
            #try:
            reward = self._reward_func(content,sol,**kwargs)
            #except Exception as e:
            #    print(f"ERROR: reward error when calcuating {self.__name__}, Exception:",e)
            rewards.append(reward)
            # TODO: fix debug log
        return rewards

empty_reward_model=SimpleNamespace(reward_model=None,processor=None,tokenizer=None)

# for lbl2img
class CLIPDistanceReward(GRPORewardFunc):
    def __init__(self,quantize=False,clone_reward=None,**kwargs):
        super().__init__(**kwargs)
        if clone_reward is not None:
            self.reward_model,self.processor,self.tokenizer=clone_reward.reward_model,clone_reward.processor,clone_reward.tokenizer
        else:
            from reward_utils.clip_score import init_clip
            self.reward_model,self.processor,self.tokenizer=init_clip(device=self.model_device)
        self.quantize=quantize
    
    def __call__(self, prompts=None, completions=None, **kwargs):
        if "text" in prompts[0]:
            solutions=[p["text"] for p in prompts]
        elif "class_labels" in prompts[0]:
            solutions=[p["class_labels"] for p in prompts]
        if type(completions) is list: # [[resolution 1],[resolution 2],...]
            completions = completions[-1]
        contents = [completion for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol in zip(contents, solutions):
            reward = 0.0
            #try:
            reward = self._reward_func(content,sol,**kwargs)
            #except Exception as e:
            #    print(f"ERROR: reward error when calcuating {self.__name__}, Exception:",e)
            rewards.append(reward)
            # TODO: fix debug log
        return rewards

    def process_image(self, img_tensor):
        if type(img_tensor) is torch.Tensor:
            img=to_pil_image(img_tensor)
        else:
            img=img_tensor
        return self.processor(images=img, return_tensors="pt") # dict["pixel_values"]
    def process_txt(self, txt_s):
        return self.tokenizer(txt_s, padding=True, return_tensors='pt') # dict["input_ids","attention_mask"]

    def _reward_func(self,cnt,lbl,**kwargs):
        from reward_utils.clip_score import calculate_one_clip_score as clip_score
        from reward_utils.imagenet_names import IMAGENET_NAMES as names
        if type(lbl) is not int:
            lbl=int(lbl)
        lbl_txt=f"a photo of {names[lbl]}"
        #print(lbl_txt)
        #print(cnt.shape)
        clip_img_tensor=self.process_image(cnt)
        clip_txt_tensor=self.process_txt(lbl_txt)
        if self.reward_model is None:
            score,delay_model=clip_score(clip_img_tensor,clip_txt_tensor,"img","txt")
            self.reward_model,self.preprocessor=delay_model
        else:
            score,_=clip_score(clip_img_tensor,clip_txt_tensor,"img","txt",self.reward_model)
        
        if self.quantize:
            score=discrete_reward_map(score,[0,0.3,0.6,1],[-1,0.5,1,1.5])
        else:
            score=5*score
        return score # we need a postive max reward

class CLIPTextReward(GRPORewardFunc):
    def __init__(self,quantize=False,clone_reward=None,**kwargs):
        super().__init__(**kwargs)
        if clone_reward is not None:
            self.reward_model,self.processor,self.tokenizer=clone_reward.reward_model,clone_reward.processor,clone_reward.tokenizer
        else:
            from reward_utils.clip_score import init_clip
            self.reward_model,self.processor,self.tokenizer=init_clip(device=self.model_device)
        self.quantize=quantize
    
    def __call__(self, prompts=None, completions=None, **kwargs):
        if "text" in prompts[0]:
            solutions=[p["text"] for p in prompts]
        elif "class_labels" in prompts[0]:
            solutions=[p["class_labels"] for p in prompts]
        if type(completions) is list: # [[resolution 1],[resolution 2],...]
            completions = completions[-1]
        contents = [completion for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol in zip(contents, solutions):
            reward = 0.0
            #try:
            reward = self._reward_func(content,sol,**kwargs)
            #except Exception as e:
            #    print(f"ERROR: reward error when calcuating {self.__name__}, Exception:",e)
            rewards.append(reward)
            # TODO: fix debug log
        return rewards

    def process_image(self, img_tensor):
        if type(img_tensor) is torch.Tensor:
            img=to_pil_image(img_tensor)
        else:
            img=img_tensor
        return self.processor(images=img, return_tensors="pt") # dict["pixel_values"]
    def process_txt(self, txt_s):
        return self.tokenizer(txt_s,truncation=True,padding=True, return_tensors='pt') # dict["input_ids","attention_mask"]

    def _reward_func(self,cnt,lbl,**kwargs):
        from reward_utils.clip_score import calculate_one_clip_score as clip_score
        if type(lbl) is not str:
            print(lbl)
        lbl_txt=lbl
        #print(lbl_txt)
        #print(cnt.shape)
        clip_img_tensor=self.process_image(cnt)
        clip_txt_tensor=self.process_txt(lbl_txt)
        if self.reward_model is None:
            score,delay_model=clip_score(clip_img_tensor,clip_txt_tensor,"img","txt")
            self.reward_model,self.preprocessor=delay_model
        else:
            score,_=clip_score(clip_img_tensor,clip_txt_tensor,"img","txt",self.reward_model)
        
        if self.quantize:
            score=discrete_reward_map(score,[0,0.3,0.6,1],[-1,0.5,1,1.5])
        else:
            score=5*score
        return score # we need a postive max reward

class HPSV21DistanceReward(CLIPDistanceReward):
    def __init__(self,clone_reward=None,**kwargs):
        super().__init__(clone_reward=empty_reward_model,**kwargs)
        if clone_reward is not None:
            self.reward_model,self.preprocessor,self.tokenizer=clone_reward.reward_model,clone_reward.preprocessor,clone_reward.tokenizer
        else:
            from reward_utils.hpsv2 import initialize_model
            self.reward_model,self.preprocessor,self.tokenizer=initialize_model(device=self.model_device,hps_version="v2.1")
    def _reward_func(self, cont, lbl, **kwargs):
        from reward_utils.hpsv2 import score as hpsv2_score
        from reward_utils.imagenet_names import IMAGENET_NAMES as names
        if type(lbl) is not int:
            lbl=int(lbl)
        lbl_txt=f"a photo of {names[lbl]}"
        if type(cont) is torch.Tensor:
            pil_img=to_pil_image(cont)
        else:
            pil_img=cont
        if self.reward_model is None:
            score,delay_model=hpsv2_score(img_or_path=pil_img,prompt=lbl_txt)
            self.reward_model,self.preprocessor,self.tokenizer=delay_model
        else:
            score,_=hpsv2_score(
                img_or_path=pil_img,prompt=lbl_txt,
                model=self.reward_model,
                preprocessor=self.preprocessor,
                tokenizer=self.tokenizer,
                )
        if self.quantize:
            score=discrete_reward_map(score,[0,0.3,0.6,1],[-1,0.5,1,1.5])
        else:
            score=5*score
        return score # we need a postive max reward

class HPSV21TextReward(CLIPTextReward):
    def __init__(self,clone_reward=None,**kwargs):
        super().__init__(clone_reward=empty_reward_model,**kwargs)
        if clone_reward is not None:
            self.reward_model,self.preprocessor,self.tokenizer=clone_reward.reward_model,clone_reward.preprocessor,clone_reward.tokenizer
        else:
            from reward_utils.hpsv2 import initialize_model
            self.reward_model,self.preprocessor,self.tokenizer=initialize_model(device=self.model_device,hps_version="v2.1")
    def _reward_func(self, cont, lbl, **kwargs):
        from reward_utils.hpsv2 import score as hpsv2_score
        from reward_utils.imagenet_names import IMAGENET_NAMES as names
        if type(lbl) is not str:
            print(lbl)
        lbl_txt=lbl
        if type(cont) is torch.Tensor:
            pil_img=to_pil_image(cont)
        else:
            pil_img=cont
        if self.reward_model is None:
            score,delay_model=hpsv2_score(img_or_path=pil_img,prompt=lbl_txt)
            self.reward_model,self.preprocessor,self.tokenizer=delay_model
        else:
            score,_=hpsv2_score(
                img_or_path=pil_img,prompt=lbl_txt,
                model=self.reward_model,
                preprocessor=self.preprocessor,
                tokenizer=self.tokenizer,
                )
        if self.quantize:
            score=discrete_reward_map(score,[0,0.3,0.6,1],[-1,0.5,1,1.5])
        else:
            score=5*score
        return score # we need a postive max reward

class QwenLabelReward(CLIPDistanceReward):
    def __init__(self,mode="transformers",clone_reward=None,**kwargs): # can use vllm and transformers
        super().__init__(clone_reward=empty_reward_model,**kwargs)
        self.mode=mode
        if mode=="transformers":
            if clone_reward is not None:
                self.pipe=clone_reward.pipe
            else:
                from reward_utils.qwen_vl_reward import init_pipeline
                self.pipe=init_pipeline(device=self.model_device)
        elif mode=="vllm":
            if clone_reward is not None:
                self.client=clone_reward.client
            else:
                from reward_utils.qwen_vl_reward import init_client
                self.client=init_client(port="20004")
        else:
            raise ValueError(f"unkown qwenlabel reward mode {mode}")
    def _reward_func(self,cnt,lbl,**kwargs):
        if self.mode=="vllm":
            from reward_utils.qwen_vl_reward import question_image_score_with_lbl_vllm as lbl_score
        elif self.mode=="transformers":
            from reward_utils.qwen_vl_reward import question_image_score_with_lbl_pipe as lbl_score
        from reward_utils.imagenet_names import IMAGENET_NAMES as names
        if type(lbl) is not int:
            lbl=int(lbl)
        if type(cnt) is torch.Tensor:
            pil_img=to_pil_image(cnt)
        else:
            pil_img=cnt
        lbl_txt=f"a photo of {names[lbl]}"
        fail_cnt=0
        score=None
        worker=self.client if self.mode=="vllm" else self.pipe
        while(fail_cnt<3):
            try:
                score_dict,_=lbl_score(pil_img,lbl_txt,worker)
                score=float(score_dict["score"])
                break
            except Exception as e:
                print(f"{self.__name__} report exception {e}, retrying... {fail_cnt}")
                score=0
                fail_cnt+=1
        return score/4 # we need a postive max reward

class QwenTextReward(CLIPTextReward):
    def __init__(self,mode="transformers",clone_reward=None,**kwargs): # can use vllm and transformers
        super().__init__(clone_reward=empty_reward_model,**kwargs)
        self.mode=mode
        if mode=="transformers":
            if clone_reward is not None:
                self.pipe=clone_reward.pipe
            else:
                from reward_utils.qwen_vl_reward import init_pipeline
                self.pipe=init_pipeline(device=self.model_device)
        elif mode=="vllm":
            if clone_reward is not None:
                self.client=clone_reward.client
            else:
                from reward_utils.qwen_vl_reward import init_client
                self.client=init_client(port="20004")
        else:
            raise ValueError(f"unkown qwenlabel reward mode {mode}")
    def _reward_func(self,cnt,lbl,**kwargs):
        if self.mode=="vllm":
            from reward_utils.qwen_vl_reward import question_image_score_with_lbl_vllm as lbl_score
        elif self.mode=="transformers":
            from reward_utils.qwen_vl_reward import question_image_score_with_lbl_pipe as lbl_score
        from reward_utils.imagenet_names import IMAGENET_NAMES as names
        if type(lbl) is not str:
            print(lbl)
        if type(cnt) is torch.Tensor:
            pil_img=to_pil_image(cnt)
        else:
            pil_img=cnt
        lbl_txt=lbl
        fail_cnt=0
        score=None
        worker=self.client if self.mode=="vllm" else self.pipe
        while(fail_cnt<3):
            try:
                score_dict,_=lbl_score(pil_img,lbl_txt,worker)
                score=float(score_dict["score"])
                break
            except Exception as e:
                print(f"{self.__name__} report exception {e}, retrying... {fail_cnt}")
                score=0
                fail_cnt+=1
        return score/4 # we need a postive max reward

class QwenFakeDiscrimReward(QwenTextReward):
    def _reward_func(self,cnt,lbl,**kwargs):
        if self.mode=="vllm":
            raise NotImplementedError("vllm not implemented")
        elif self.mode=="transformers":
            from reward_utils.qwen_vl_reward import question_image_score_prompt_key_pipe as lbl_score
        from reward_utils.imagenet_names import IMAGENET_NAMES as names
        if type(cnt) is torch.Tensor:
            pil_img=to_pil_image(cnt)
        else:
            pil_img=cnt
        lbl_txt="fake_judge"
        fail_cnt=0
        score=None
        worker=self.client if self.mode=="vllm" else self.pipe
        while(fail_cnt<3):
            try:
                score_dict,_=lbl_score(pil_img,lbl_txt,worker)
                score=float(score_dict["score"])
                break
            except Exception as e:
                print(f"{self.__name__} report exception {e}, retrying... {fail_cnt}")
                score=0
                fail_cnt+=1
        return score/4 # we need a postive max reward

class QwenWeirdDiscrimReward(QwenTextReward):
    def _reward_func(self,cnt,lbl,**kwargs):
        if self.mode=="vllm":
            raise NotImplementedError("vllm not implemented")
        elif self.mode=="transformers":
            from reward_utils.qwen_vl_reward import question_image_score_prompt_key_pipe as lbl_score
        from reward_utils.imagenet_names import IMAGENET_NAMES as names
        if type(cnt) is torch.Tensor:
            pil_img=to_pil_image(cnt)
        else:
            pil_img=cnt
        lbl_txt="weird_judge"
        fail_cnt=0
        score=None
        worker=self.client if self.mode=="vllm" else self.pipe
        while(fail_cnt<3):
            try:
                score_dict,_=lbl_score(pil_img,lbl_txt,worker)
                score=float(score_dict["score"])
                break
            except Exception as e:
                print(f"{self.__name__} report exception {e}, retrying... {fail_cnt}")
                score=0
                fail_cnt+=1
        return score/4 # we need a postive max reward

#Flow GRPO rewards

class GenEvalTextRewardBatch(GRPORewardFunc):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from reward_utils.geneval_client import create_session
        self.reward_model=create_session()
    
    def __call__(self, prompts=None, completions=None, **kwargs):
        if "geneval_json" in prompts[0]:
            solutions=[p["geneval_json"] for p in prompts]
        else:
            # no geneval metadatas
            return [0]*len(prompts)
        if type(completions) is list: # [[resolution 1],[resolution 2],...]
            completions = completions[-1]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        rewards = self._reward_func(completions,solutions)
        return rewards

    def _reward_func(self,cnt,sol,**kwargs):
        from reward_utils.geneval_client import geneval_score
        if self.reward_model is None:
            rewards,self.reward_model=geneval_score(cnt,sol)
        else:
            rewards,_=geneval_score(cnt,sol,self.reward_model)
        return rewards

class UnifiedRewardBatch(GRPORewardFunc):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from reward_utils.unireward import create_session
        self.reward_model=create_session()
    
    def __call__(self, prompts=None, completions=None, **kwargs):
        if "text" in prompts[0]:
            solutions=[p["text"] for p in prompts]
        else:
            # no geneval metadatas
            return [0]*len(prompts)
        if type(completions) is list: # [[resolution 1],[resolution 2],...]
            completions = completions[-1]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        rewards = self._reward_func(completions,solutions)
        return rewards

    def _reward_func(self,cnt,sol,**kwargs):
        from reward_utils.unireward import unified_reward
        if self.reward_model is None:
            rewards,self.reward_model=unified_reward(cnt,sol)
        else:
            rewards,_=unified_reward(cnt,sol,self.reward_model)
        return rewards

class PickscoreRewardBatch(UnifiedRewardBatch):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from reward_utils.pickscore import create_model
        self.reward_model=create_model()

    def _reward_func(self,cnt,sol,**kwargs):
        from reward_utils.pickscore import pickscore as score_fun
        if self.reward_model is None:
            rewards,self.reward_model=score_fun(cnt,sol)
        else:
            rewards,_=score_fun(cnt,sol,self.reward_model)
        return rewards

class ImageRewardBatch(UnifiedRewardBatch):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from reward_utils.imagereward import create_model
        self.reward_model=create_model()

    def _reward_func(self,cnt,sol,**kwargs):
        from reward_utils.imagereward import imagereward_score as score_fun
        if self.reward_model is None:
            rewards,self.reward_model=score_fun(cnt,sol)
        else:
            rewards,_=score_fun(cnt,sol,self.reward_model)
        return rewards

# for NR IQA
class MANIQAReward(GRPORewardFunc):
    def __init__(self,quantize=False,**kwargs):
        super().__init__(**kwargs)
        from reward_utils.maniqa import init_model
        self.reward_model=init_model(device=self.model_device)
        self.quantize=quantize

    def process_image(self, img_tensor):
        if type(img_tensor) is torch.Tensor:
            img=to_pil_image(img_tensor)
        elif type(img_tensor) is np.ndarray:
            img=Image.from_array(img_tensor)
        else:
            img=img_tensor
        return img

    def _reward_func(self, cont, sol, **kwargs):
        from reward_utils.maniqa import calc_one_img
        if type(cont) is list:
            cont=cont[-1]
        #self.reward_model.to(cont.device)
        if type(cont) is torch.Tensor:
            cont=self.process_image(cont)
        iqa_score=calc_one_img(self.reward_model,cont)
        #reward=discrete_reward_map(mse,[0,50,100,200,500,1000],[-1,2,1.5,1,0.5,0.1,0])
        reward=iqa_score
        if reward<0:
            reward=0
            raise ValueError(f"Invalid iqa_score value {reward}")
        if self.quantize:
            reward=discrete_reward_map(reward,[0,0.3,0.6,1],[-1,0.5,1,1.5])
        else:
            reward=2*reward
        return reward

class DeQARewardBatch(GRPORewardFunc):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from reward_utils.deqa_client import create_session
        self.reward_model=create_session()
    
    def __call__(self, prompts=None, completions=None, **kwargs):
        if type(completions) is list: # [[resolution 1],[resolution 2],...]
            completions = completions[-1]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        rewards = self._reward_func(completions)
        return rewards

    def _reward_func(self,cnt,**kwargs):
        from reward_utils.deqa_client import deqa_score as score_fun
        if self.reward_model is None:
            rewards,self.reward_model=score_fun(cnt)
        else:
            rewards,_=score_fun(cnt,self.reward_model)
        return rewards

class AestheticRewardBatch(DeQARewardBatch):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from reward_utils.aesthscore import create_model
        self.reward_model=create_model()

    def _reward_func(self,cnt,**kwargs):
        from reward_utils.aesthscore import aesthetic_score as score_fun
        if self.reward_model is None:
            rewards,self.reward_model=score_fun(cnt)
        else:
            rewards,_=score_fun(cnt,self.reward_model)
        return rewards

# for img2img

class MSEReward(GRPORewardFunc):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from torch.nn import MSELoss
        self.reward_model=MSELoss()
    def _reward_func(self, cont, sol, **kwargs):
        if type(cont) is list:
            cont=cont[-1]
        self.reward_model.to(cont.device)
        mse=self.reward_model(cont,sol)
        #reward=discrete_reward_map(mse,[0,50,100,200,500,1000],[-1,2,1.5,1,0.5,0.1,0])
        reward=(1-mse)
        if reward<0:
            reward=0
            raise ValueError(f"Invalid mse value {mse}")
        return 2*reward

class LPIPSReward(GRPORewardFunc):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        import lpips
        self.reward_model=lpips.LPIPS(net='vgg')
        
    def _reward_func(self, cont, sol, **kwargs):
        if type(cont) is list:
            cont=cont[-1]
        self.reward_model.to(cont.device)
        lpips_score=self.reward_model(cont,sol)
        reward=discrete_reward_map(lpips_score,[0,50,100,200,500,1000],[-1,2,1.5,1,0.5,0.1,0])
        if reward<0:
            reward=0
            raise ValueError(f"Invalid lpips score {lpips_score}")
        return reward

class FIDReward(GRPORewardFunc):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from reward_utils.fid import init_model,load_imagenet_stat
        self.reward_model=init_model(self.model_device,2048)
        if "group" in kwargs:
            self.group_num=kwargs["group"]
        else:
            self.group_num=2
        self.ref_stat=load_imagenet_stat()

    def __call__(self, prompts=None, completions=None, **kwargs): # split to groups and calc each group fid
        if type(completions) is list: # [[resolution 1],[resolution 2],...]
            completions = completions[-1]
        assert type(completions) is torch.Tensor, str(type(completions))
        group_size=completions.shape[0]//self.group_num
        contents = torch.split(completions,group_size,dim=0)
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for g in contents:
            #try:
            g_reward = self._reward_func(g,**kwargs)
            #except Exception as e:
            #    print(f"ERROR: reward error when calculating {self.__name__}, Exception:",e)
            rewards.extend([g_reward]*g.shape[0])
            # TODO: fix debug log
        assert len(rewards)==completions.shape[0]
        return rewards

    def _reward_func(self, cont_g, **kwargs):
        from reward_utils.fid import calculate_fid 
        reward=0
        if self.reward_model is None:
            fid_value,self.reward_model = calculate_fid(cont_g,refs_stat=self.ref_stat)
        else:
            fid_value,_ = calculate_fid(cont_g,refs_stat=self.ref_stat,model=self.reward_model)
        reward=discrete_reward_map(fid_value,[0,2,5,10,20,50,75,100,125,175,200,300,500,1000],
                                            [-1,2,1.8,1.5,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0.05,0])
        #reward=fid_value
        if reward<0:
            reward=0
            raise ValueError(f"Invalid fid value {fid_value}")
        return reward

class CMMDReward(GRPORewardFunc):
    def _reward_func(self, cont, sol, **kwargs):
        reward=0
        return reward

class PRDReward(GRPORewardFunc):
    def _reward_func(self, cont, sol, **kwargs):
        reward=0
        return reward

class CLIP_FIDReward(GRPORewardFunc):
    def _reward_func(self, cont, sol, **kwargs):
        reward=0
        return reward


class VQAScoreReward(GRPORewardFunc):
    def __init__(self, model="llava-v1.6-13b", log_dir='images/ours_ar_timestep', log_file="./log_testcurr.jsonl", 
                 use_fine_grained=True, use_calibration=False, clone_reward=None, **kwargs):
        super().__init__(**kwargs)
        
        assert not use_calibration or use_fine_grained, "use_calibration can only be True when use_fine_grained=True"
        self.use_fine_grained = use_fine_grained
        self.use_calibration = use_calibration
        
        if clone_reward is not None:
            self.reward_model = clone_reward.reward_model
        else:
            try:
                self.reward_model = t2v_metrics.VQAScore(model=model, device='cuda', cache_dir="/mnt/sdb3/runhaofu/hugging_face/hub")
                if self.model_device:
                    self.reward_model = self.reward_model.to(self.model_device)
                self.reward_model.eval()
            except Exception as e:
                print(f"Failed to initialize VQAScore model: {e}")
                self.reward_model = None

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = log_file
        # 初始化日志文件
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                pass
        except:
            print(f"Warning: Could not initialize log file {self.log_file}")

    def __call__(self, prompts=None, completions=None, **kwargs):
        if self.reward_model is None:
            print("VQAScore model not available.")
            exit(0)
            
        # 提取图像路径和场景数据
        if self.use_fine_grained:
            # 需要从prompts中提取qa数据
            if "qa" not in prompts[0]:
                print("Warning: Fine-grained mode requires 'qa' data in prompts")
                return [0.0] * len(prompts)
            scenes = [{"qa": p["qa"], "prompt": p.get("text", ""), "difficulty": p.get("difficulty", "unknown")} 
                     for p in prompts]
        else:
            # 使用prompt模式
            if "text" not in prompts[0]:
                print("Warning: Prompt mode requires 'prompt' data in prompts")
                return [0.0] * len(prompts)
            scenes = [{"prompt": p["text"], "difficulty": p.get("difficulty", "unknown")} 
                     for p in prompts]
        
        if type(completions) is list:
            completions = completions[-1]
        
        return self._reward_func(completions, scenes, **kwargs)

    def _reward_func(self, images: List, scenes: List[Dict], **kwargs):
        if self.reward_model is None:
            exit(0)
            
        vqa_score_list = []
        for image, scene in zip(images, scenes):
            try:
                if self.use_fine_grained:
                    # Use fine-grained questions from qa
                    questions = []
                    dependencies = []
                    question_types = []

                    # Process each category
                    categories = ["object", "count", "attribute", "relation"]
                    for category in categories:
                        if category in scene["qa"]:
                            for item in scene["qa"][category]:
                                questions.append(item["question"])
                                dependencies.append(item["dependencies"])
                                question_types.append(category)

                    if not questions:
                        print("Warning: No questions found in qa data")
                        vqa_score_list.append(0.0)
                        continue

                    support_data = {"questions": questions, "dependencies": dependencies, "question_types": question_types}
                    
                    # 处理图像输入
                    image_path = to_temp_path(image, img_name=scene['prompt'], log_dir=self.log_dir)
                    image_input = [image_path] 
                    
                    raw_vqa_score = self.reward_model(image_input, support_data["questions"])[0].tolist()

                    
                    if self.use_calibration:
                        # Apply calibration and record calibrated scores
                        vqa_score = []
                        sum_score = 0
                        for score, dependency, question_type in zip(raw_vqa_score, dependencies, question_types):
                            if question_type == "object":
                                calibrated_score = score
                            elif question_type == "attribute" or question_type == "count":
                                try:
                                    calibrated_score = score * raw_vqa_score[dependency[0] - 1]
                                except IndexError as e:
                                    print(f"vqascore:{raw_vqa_score}, type:{question_types}, dependency:{dependency}")
                                    calibrated_score = score
                            elif question_type == "relation":
                                try:
                                    calibrated_score = score * (min(raw_vqa_score[dependency[0] - 1], raw_vqa_score[dependency[1] - 1]))
                                except IndexError as e:
                                    print(f"vqascore:{raw_vqa_score}, type:{question_types}, dependency:{dependency}")
                                    calibrated_score = score
                            else:
                                raise ValueError("Not implemented question type error")
                            
                            vqa_score.append(calibrated_score)
                            sum_score += calibrated_score
                        
                        avg_vqa_score = sum_score / len(questions)
                    else:
                        # No calibration, use raw scores directly
                        vqa_score = raw_vqa_score
                        avg_vqa_score = sum(vqa_score) / len(questions)
                    
                else:
                    # Use prompt
                    questions = [scene["prompt"]]
                    if isinstance(image, str):
                        image_input = [image]
                    else:
                        image_input = [image]
                    vqa_score = self.reward_model(image_input, questions).item()
                    avg_vqa_score = vqa_score

                vqa_score_list.append(avg_vqa_score)
                
                # 记录日志
                try:
                    log_data = {
                        "image_path": image_path,
                        "difficulty": scene["difficulty"],
                        "prompt": scene["prompt"] if "prompt" in scene else "",
                        "questions": questions if self.use_fine_grained else [scene["prompt"]],
                        "vqa_score": vqa_score if isinstance(vqa_score, list) else [vqa_score],
                        "avg_vqa_score": avg_vqa_score,
                        "use_fine_grained": self.use_fine_grained,
                        "use_calibration": self.use_calibration,
                    }

                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Warning: Failed to write log: {e}")
                    
            except Exception as e:
                print(f"Error processing VQAScore for image: {e}")
                vqa_score_list.append(0.0)

        return torch.tensor(vqa_score_list, dtype=torch.float32, device=self.model_device)