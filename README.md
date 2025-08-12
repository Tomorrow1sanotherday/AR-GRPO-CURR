# AR-GRPO: Training Autoregressive Image Generation Models via Reinforcement Learning

![Last Commit](https://img.shields.io/github/last-commit/Kwai-Klear/AR-GRPO)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Kwai-Klear/AR-GRPO/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

**[[arXiv]](https://arxiv.org/pdf/2508.06924) | [[Codes]](https://github.com/Kwai-Klear/AR-GRPO) | [🤗 Huggingface]** <br> 
Shihao Yuan, Yahui Liu♥, Yang Yue, Jingyuan Zhang, Wangmeng Zuo♥, Qi Wang, Fuzheng Zhang, Guorui Zhou <br>
*♥ Corresponding author*

## 1. Abstract

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
RLbased optimization for AR image generation and opening new avenues for controllable and
high-quality image synthesis. 


**Source codes and models are coming soon ...**

## 2. Citation
```latex
@article{yuan2025argrpo,
  title={AR-GRPO: Training Autoregressive Image Generation Models via Reinforcement Learning},
  author={Yuan, Shihao and Liu, Yahui and Yue, Yang and Zhang, Jingyuan and Zuo, Wangmeng and Wang, Qi and Zhang, Fuzheng and Zhou, Guorui},
  journal={arXiv preprint arXiv:2508.06924},
  year={2025}
}
```
