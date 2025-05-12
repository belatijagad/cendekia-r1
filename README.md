# Cendekia R1

*An attempt to provide reasoning capability for smaller language model, specialized for Bahasa Indonesia. This project is intended as the final project of my Machine/Reinforcement Learning course in Universitas Indonesia.*

## Overview

The main goal of this repository is to reproduce the reasoning capability shown in language model reasoning papers.

## Installation

For setup, run:

```shell
uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install --upgrade pip && uv pip install -r requirements.txt
```

Install prebuilt flash-attention
```
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.5/flash_attn-2.7.4.post1+cu124torch2.6-cp310-cp310-linux_x86_64.whl
```

When reopening project, run:

```shell
source .venv/bin/activate
```

## Run the Code
```bash
accelerate launch --num_processes 3 --config_file deepspeed_zero3.yaml main.py --config config.yaml
```

## Model
The trained model is stored in this [Huggingface Models](https://huggingface.co/belatijagad/cendekia-r1-zero-qwen-2.5-1.5b-instruct)

## Dataset
The processed dataset is stored in this [Kaggle Dataset](https://www.kaggle.com/datasets/belati/cendekia-reasoning-math)

## IF THE CODE DOESN'T WORK
Please refer to this [training code](https://www.kaggle.com/code/belati/cendekia-grpo) and [inference code](https://www.kaggle.com/code/belati/cendekia-experiment).