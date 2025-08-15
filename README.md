# AR-GRPO: Training Autoregressive Image Generation Models via Reinforcement Learning

![Last Commit](https://img.shields.io/github/last-commit/Kwai-Klear/AR-GRPO)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Kwai-Klear/AR-GRPO/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

**[[arXiv]](https://arxiv.org/pdf/2508.06924) | [[Codes]](https://github.com/Kwai-Klear/AR-GRPO) | [[🤗 Huggingface]](https://huggingface.co/collections/CSshihao/ar-grpo-689c970f4c848f01a162352a)** <br> 
Shihao Yuan, Yahui Liu♥, Yang Yue, Jingyuan Zhang, Wangmeng Zuo♥, Qi Wang, Fuzheng Zhang, Guorui Zhou <br>
*♥ Corresponding author*

## Abstract

Inspired by the success of reinforcement learning (RL) in refining large language models (LLMs), 
we propose AR-GRPO, an approach to integrate online RL training into autoregressive (AR)
image generation models. We adapt the Group Relative Policy Optimization (GRPO) algorithm
to refine the vanilla autoregressive models’ outputs by carefully designed reward functions that
evaluate generated images across multiple quality dimensions, including perceptual quality,
realism, and semantic fidelity. We conduct comprehensive experiments on both class-conditional
(i.e., class-to-image) and text-conditional (i.e., text-to-image) image generation tasks, demonstrating
that our RL-enhanced framework significantly improves both the image quality and human
preference of generated images compared to the standard AR baselines. Our results show
consistent improvements across various evaluation metrics, establishing the viability of 
RL-based optimization for AR image generation and opening new avenues for controllable and
high-quality image synthesis. 


## Installation
Create a new Python environment with `conda`,
```shell
conda env create -f conda_environment.yaml
```
If you wish to install dependencies manually using `pip`, use following commands,
```shell
conda create -n ar-grpo python==3.10
conda activate ar-grpo
pip install -r requirements.txt
```

## Evaluation
### C2I evaluation
C2I evaluation with FID and other metrics is from [OpenAI](https://github.com/openai/guided-diffusion/tree/main/evaluations). For environment requirments, Please refer to [requiremtes.txt](https://github.com/openai/guided-diffusion/blob/main/evaluations/requirements.txt). We recommand creating a new environment for evaluation to prevent conflicts of dependencies. 
Also, you should download [ImageNet reference batch 256x256](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) before evaluation.

After environment installation, you can use `test.sh` for C2I inference on ImageNet.
```shell
bash test.sh MODEL_NAME GPU_RANK STEP CFG
```
For example,
```shell
bash test.sh my_c2i_model 0 100 2.0
```
You would find a `.npz` file in `test_results/`, then use `fid_evaluation.py` to collect metrics.
```shell
python fid_evaluation.py IMAGENET_REF_PATH/VIRTUAL_imagenet256_labeled.npz test_results/YOUR_RESULTS.npz
```

### T2I evaluation
#### GenEval
Create a new environment and download pretrained models according to [GenEval](https://github.com/djghosh13/geneval). You may need to change pretrained model path in `benchmark/geneval/eval.sh` to your downloaded model path. Then run `test_t2i.sh` to generate images from GenEval test prompts.
```shell
bash test_t2i.sh MODEL_NAME GPU_RANK STEP CFG
```
Then run `eval.sh` in GenEval to collect results.
```shell
cd ./benchmark/geneval
bash eval.sh test_results/GENERATED_IMAGES GPR_RANK
```

#### Other Metrics
For metrics like CLIP Score, HPSv2 and others, we implement them in a form of reward class in `img_gen_grpo_rewards.py`, so they can be used both in training and evaluation. 
For CLIP Score, HPSv2, PickScore, ImageReward and Aesthetic Score, their dependencies are included in `requirements.txt` already, so no other operations needed. 
For DeQA, please refer to [reward-server](https://github.com/yifan123/reward-server) to set a environment and deploy a backend server and modify address and port in `reward_utils/deqa_client.py`.
For UniReward, please create a new `sglang` environment and start a server as below,
```shell
conda create -n sglang python=3.10
conda activate sglang
pip install "sglang[all]"

python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key unireward --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.5
```
If you only want to test certain scores, you could modify `test_t2i_rewards.py` to remove unwanted metrics. Once you have all metrics you want, run following commands to generate images from DrawBench prompts.
```shell
bash test_t2i_rewards_drawbench.sh MODEL_NAME GPU_RANK STEP CFG
```
The results should be printed to the screen after evaluation is complete.

## Training
### Data Preparation
For T2I, the training prompts are already in `dataset/coco_captions_30000.json`.
For C2I, please download [ImageNet](http://image-net.org/), and change the `data_rt` in training scripts.

### GRPO Training
Download pretrained models from [LlamaGen](https://github.com/FoundationVision/LlamaGen) if you want to train your own model.
Reward models requires `openai/clip-vit-large-patch14` and `Qwen/Qwen2.5-VL-3B-Instruct`. Usually they should be downloaded automaticly from `huggingface.co`. If the auto download fails, you could download them manaully and change the path for corresponding rewards in `reward_utils`.

Change any parameters as you wish in training scripts.
For C2I training, run following command,
```shell
bash run_c2i_train.sh
```
For T2I training, run following command,
```shell
bash run_t2i_train.sh
```

## Citation
```latex
@article{yuan2025argrpo,
 title={AR-GRPO: Training Autoregressive Image Generation Models via Reinforcement Learning},
 author={Yuan, Shihao and Liu, Yahui and Yue, Yang and Zhang, Jingyuan and Zuo, Wangmeng and Wang, Qi and Zhang, Fuzheng and Zhou, Guorui},
 journal={arXiv preprint arXiv:2508.06924},
 year={2025}
}
```
