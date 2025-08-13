import transformers
from modified_grpo_trainer import GRPOTrainer_M
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
from trl import GRPOTrainer

from reward_utils.imagenet_names import IMAGENET_NAMES as names

class ImgGenGRPOTrainer(GRPOTrainer_M):
    def __init__(self,**kwargs):
        if "exit_step" in kwargs:
            self.exit_step=kwargs.pop("exit_step")
        super().__init__(**kwargs)
    
    def log_images(self,outputs,labels=None):
        if not hasattr(self,"tf_swr"):
            return
        log_step=min(self.state.max_steps,self.exit_step)//10
        if dist.get_rank()==0 and self.state.global_step%log_step==0:
            if type(outputs) is not list:
                outputs=[outputs]
            imgs=[make_grid(i, nrow=8, padding=0, pad_value=1.0) for i in outputs]
            for idx,i in enumerate(imgs):
                self.tf_swr.add_image(f"resolution_{idx}",i,self.state.global_step)
            #if type(labels) is list or type(labels) is torch.Tensor:
            #    for idx,l in enumerate(labels):
            #        self.tf_swr.add_text(f"labels_{idx}",names[l.item()]+f"({l.item()})",self.state.global_step)
            #else:
            #   self.tf_swr.add_text("label",names[labels]+f"({labels})",self.state.global_step)

    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args,**kwargs)
        #print("step",self.state.global_step)
        if self.state.global_step>=self.exit_step:
            exit(0)