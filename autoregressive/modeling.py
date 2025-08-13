import torch
import transformers
import torch.nn as nn
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel, AutoConfig, PretrainedConfig, GenerationConfig, GenerationMixin, PreTrainedTokenizer
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace
from copy import deepcopy
from tokenizer_lg.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from language.t5 import T5Embedder
from autoregressive.models.generate import generate as model_generate


GLOBAL_MODEL_TYPE="ImgGen_HF"

class ImgGenTextImageProcessor(PreTrainedTokenizer): # for grpo processing class to avoid AutoTokenizer, in charge of encoding and decoding
    pad_token="</s>"
    pad_token_id=99999999
    bos_token="<s>"
    bos_token_id=0
    eos_token="</s>"
    eos_token_id=99999999 # should be bigger than vqvae's vocabulary
    def __init__(self, text_encoder=None, image_tokenizer=None, codebook_embed_dim=None, **kwargs):
        self.text_encoder=text_encoder
        self.image_tokenizer=image_tokenizer
        self.codebook_embed_dim=codebook_embed_dim

    def __call__(self, *args, **kwargs): # encoding
        inputs=kwargs["text"] # just for not breaking trl grpo trainer, call it text
        if "image" in inputs[0] and inputs[0]["image"] is not None:
            images=torch.stack([i["image"] for i in inputs],dim=0)
        else:
            images=None
        if "class_labels" in inputs[0]:
            input_lbls=[i["class_labels"] for i in inputs]
            if type(input_lbls[0]) is torch.Tensor:
                input_ids=torch.stack(input_lbls,dim=0)
            else:
                input_ids=torch.tensor(input_lbls)
                if input_ids.ndim<2:
                    input_ids=input_ids.unsqueeze(-1)
            attention_mask=torch.ones_like(input_ids)
        elif "text" in inputs[0]:
            input_text=[i["text"] for i in inputs]
            #print(input_text)
            input_ids,attention_mask=self.text_encoder.text_tokenize(input_text)
        else:
            raise ValueError("No labels or texts in input")
        #print(input_ids)
        #print(input_ids.shape,input_ids.dtype)
        if self.image_tokenizer is not None and images is not None:
            image_ids=self.image_tokenizer.encode(images)[-1][-1]
        else:
            image_ids=None
        return {"input_ids":input_ids,"attention_mask":attention_mask,"image_ids":image_ids}
    
    def encode_img(self, img): # B3HW
        pass

    def decode(self, input_ids, **kwargs):
        bs=input_ids.shape[0]
        #print(input_ids.shape)
        hxw=input_ids.shape[1]
        h=int(hxw**0.5)
        w=int(hxw//h)
        gss_shape=[bs,self.codebook_embed_dim,h,w]
        res_img=self.image_tokenizer.decode_code(input_ids,gss_shape)
        return torch.clamp(res_img.add_(1).mul_(0.5),0,1)

    def batch_decode(self, input_ids,**kwargs):
        if type(input_ids) is list:
            return [ self.decode(i) for i in input_ids]
        else:
            return self.decode(input_ids)

    def save_pretrained(self,out_dir):
        pass # using vae from main model as processor

class ImgGen_HF_CONFIG(PretrainedConfig):
    model_type=GLOBAL_MODEL_TYPE
    keys_to_ignore_at_inference=[]
    self_model_config_attrs=dict( # customized for easy manage model configs, only contain default values, do not use it as real config
        vq_model="VQ-16",
        gpt_model="GPT-B",
        codebook_size=16384,
        codebook_embed_dim=8,
        image_size=384,
        downsample_size=16,
        cls_token_num=1,
        gpt_type="c2i",
        text_encoder_ckpt="",
        text_encoder="google/flan-t5-xl",
        text_feature_max_len=120,
        dropout_p=0.1,
        token_dropout_p=0.1,
        drop_path_rate=0.0,
        delay_load_text_encoder=False,
    )
    def __init__(
        self,
        **kwargs,
    ):
        # grab model configs, should be static but cant use generator in static domain
        for k in self.self_model_config_attrs:
            if k in kwargs: # we should avoid using attr name directly in cmd args, but for now we use
                setattr(self,k,kwargs.pop(k)) # real config values should be directly attributes of config class, pop kwargs for super class
            else:
                setattr(self,k,self.self_model_config_attrs[k])
        super().__init__(**kwargs)

class ImgGen_HF(PreTrainedModel,GenerationMixin):
    config_class=ImgGen_HF_CONFIG
    base_model_prefix="model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False
    default_generation_config=dict(cfg_scale=1.5,cfg_interval=-1,temperature=1.0,top_k=0,top_p=1.0,sample_logits=True,return_logits=True)
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if type(config) is dict: # completely have no idea why it is a dict, but keep it for accidents
            config=self.config_class.from_dict(config) 
        assert isinstance(config,self.config_class), f"only accept corresponding config class, not {type(config)}"
        model_config={k:getattr(config,k) for k in config.self_model_config_attrs}
        self.model_device="cuda" if torch.cuda.is_available() else "cpu"

        args=SimpleNamespace(**model_config)
        self.vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
        #self.vq_model.to(self.model_device)
        self.delay_load_text_encoder=getattr(config,"delay_load_text_encoder",False)

        self.latent_size = args.image_size // args.downsample_size
        self.gpt_model = GPT_models[args.gpt_model](
            block_size=self.latent_size ** 2,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=args.dropout_p,
            ffn_dropout_p=args.dropout_p,
            drop_path_rate=args.drop_path_rate,
            token_dropout_p=args.token_dropout_p,
        )#.to(device=self.model_device)

        #print(self.config.gpt_type)
        if self.config.gpt_type=="t2i":
            if self.delay_load_text_encoder:
                self.text_encoder=dict(
                    device=self.model_device, 
                    local_cache=False, 
                    cache_dir=args.text_encoder_ckpt, 
                    dir_or_name=args.text_encoder,
                    model_max_length=args.text_feature_max_len,
                )
            else:
                self.text_encoder = T5Embedder(
                    device=self.model_device, 
                    local_cache=False, 
                    cache_dir=args.text_encoder_ckpt, 
                    dir_or_name=args.text_encoder,
                    model_max_length=args.text_feature_max_len,
                )
        else:
            self.text_encoder=None

        self.all_models=["vq_model","gpt_model"]

        self.config=config
    
    def load_vq(self, vq_ckpt, map_location='cpu'):
        self.vq_model.load_state_dict(torch.load(vq_ckpt, map_location=map_location)["model"])
        self.vq_model.to(self.model_device)
        print(f"image tokenizer is loaded")

    def load_model(self, model_ckpt, map_location='cpu'):
        checkpoint=torch.load(model_ckpt, map_location=map_location)
        if "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "module" in checkpoint: # deepspeed
            model_weight = checkpoint["module"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            model_weight = checkpoint
            #raise Exception("please check model weight")
        self.gpt_model.load_state_dict(model_weight, strict=False)
        self.gpt_model.to(self.model_device)
        print(f"gpt model is loaded")

    def load_text_encoder(self):
        if type(self.text_encoder) is dict:
            self.text_encoder=T5Embedder(**self.text_encoder)
            print("text encoder delay loaded")


    def get_model_device(self):
        return next(self.gpt_model.parameters()).device

    # rewrite a generate function for image generation that take place generate() from GenerationMixin class originally for text generation
    def prepare_inputs_for_generation(
        self
    ):
        pass
    
    def find_prefix(self,s,l):
        for i in l:
            if s.startswith(i):
                return True
        return False

    @property
    def total_params(self):
        return sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in self.parameters())
    
    @property
    def trainable_params(self):
        return sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_model(self,excluded_list=None):
        for model in self.all_models:
            for name, param in getattr(self,model).named_parameters(): 
                param.requires_grad_(False)
        if excluded_list is not None:
            submodule_excluded={n:[] for n in self.all_models}
            for m in excluded_list:
                for mname in self.all_models:
                    if m.startswith(mname):
                        submodule_excluded[mname].append(m.replace(mname,"",1).strip("."))
            for model in self.all_models:
                for name, param in getattr(self,model).named_parameters(): 
                    if self.find_prefix(name,submodule_excluded[model]):
                        param.requires_grad_(True)

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.Tensor] = None, attention_mask=None, generation_config: Optional[GenerationConfig] = None, **kwargs
    ) -> torch.Tensor: 
        cur_generation_config=deepcopy(self.default_generation_config)
        if generation_config is not None:
            for k in cur_generation_config:
                if hasattr(generation_config,k):
                    cur_generation_config[k]=getattr(generation_config,k)
        # grab generation args from kwargs directly
        for k in kwargs:
            if k in cur_generation_config:
                cur_generation_config[k]=kwargs[k]

        #print(cur_generation_config)
        self.gpt_model.eval() # we must manually set this because of dropout layers
        if self.config.gpt_type=="t2i":
            input_embed,embed_mask=self.text_encoder.text_embedding(input_ids,attention_mask)
            #print(input_embed.shape,embed_mask.shape)
            res=model_generate(
                self.gpt_model, input_embed, self.latent_size ** 2, cond_ids=input_ids, emb_masks=embed_mask,
                **cur_generation_config
            )
        else:
            res=model_generate(
                self.gpt_model, input_ids, self.latent_size ** 2,
                **cur_generation_config
            )
        self.gpt_model.train()
        res_seq=res["input_ids"]
        res_logits=res["logits"] if "logits" in res else None
        #print(res_logits.requires_grad)
        res_output=GenerateDecoderOnlyOutput(
            sequences=res_seq,
            logits=res_logits,
        )
        res_output.output_ids=res["output_ids"]
        #res_output.cond_BD=res["cond_BD"]
        #res_output.input_token_maps=res["input_token_maps"]
        return res_output

    def forward(self,input_ids=None,attention_mask=None,logits_to_keep=None,eval_mode=True,**kwargs): 
        
        bs=input_ids.shape[0]
        max_seq_length=input_ids.shape[1]

        if hasattr(self,"cfg_scale"):
            cfg_scale=self.cfg_scale
        elif "cfg_scale" in kwargs:
            cfg_scale=kwargs["cfg_scale"]
        else:
            cfg_scale=1.0

        if eval_mode:
            self.gpt_model.eval()
        self.gpt_model.disable_caches()
        #with torch.device("cuda"):
        #    self.gpt_model.setup_caches(max_batch_size=bs, max_seq_length=max_seq_length, dtype=self.gpt_model.tok_embeddings.weight.dtype)
        # logits_to_keep got +1 in grpo trainer, so we have to -1 to match the real len of completion_ids
        cond_ids=input_ids[:,:-logits_to_keep+1]
        #print(cond_ids.shape)
        #print(cond_ids)
        img_ids=input_ids[:,-logits_to_keep+1:]
        #cond_mask=attention_mask[:,:-logits_to_keep]
        #img_mask=attention_mask[:,-logits_to_keep:]
        attn_mask=torch.tril(torch.ones(bs, max_seq_length, max_seq_length, device=input_ids.device))
        #attn_mask[:]=attn_mask-torch.eye(max_seq_length,max_seq_length, device=input_ids.device)
        #attn_mask[:,-logits_to_keep+1:,:-logits_to_keep+1]=False # cond_idx no longer needed after prefill
        if attention_mask is not None:
            attn_mask=attn_mask*attention_mask.unsqueeze(1)
        attn_mask=attn_mask.unsqueeze(1).to(torch.bool)
        #print(attn_mask)
        #print(attn_mask.shape)
        input_pos = torch.arange(0, max_seq_length, device=input_ids.device)
        #attn_mask=None
        #input_pos=None
        if cfg_scale > 1.0:
            if self.config.gpt_type == 'c2i':
                cond_null = torch.ones_like(cond_ids) * self.gpt_model.num_classes
            elif self.config.gpt_type == 't2i':
                cond_ids,cond_mask = self.text_encoder.text_embedding(cond_ids,attention_mask)
                cond_null = torch.zeros_like(cond_ids) + self.gpt_model.cls_embedding.uncond_embedding
            else:
                raise ValueError("please check model type")
            cond_ids = torch.cat([cond_ids, cond_null])
            img_ids = torch.cat([img_ids,img_ids])
            attn_mask = torch.cat([attn_mask,attn_mask])
            logits,loss=self.gpt_model(cond_idx=cond_ids, idx=img_ids, targets=None, mask=attn_mask, input_pos=input_pos)
            cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0) 
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            if self.config.gpt_type == 't2i':
                cond_ids,cond_mask = self.text_encoder.text_embedding(cond_ids,attention_mask)
            logits,loss=self.gpt_model(cond_idx=cond_ids, idx=img_ids, targets=None, mask=attn_mask, input_pos=input_pos)
        

        return CausalLMOutputWithPast(
            logits=logits[:,-logits_to_keep:],
            loss=loss,
        )
