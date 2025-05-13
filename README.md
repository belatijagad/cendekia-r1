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
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.5/flash_attn-2.7.4.post1+cu124torch2.6-cp311-cp311-linux_x86_64.whl
```

When reopening project, run:

```shell
source .venv/bin/activate
```

## Setup

Download dataset

```bash
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
kaggle datasets download belati/cendekia-reasoning-math
```

```bash
sudo apt-get install unzip
unzip cendekia-reasoning-math.zip
mkdir data
mv train_data.csv test_data.csv data
rm cendekia-reasoning-math.zip
```

## Run the Code

```
CUDA_DEVICE_ORDER="PCI_BUS_ID" PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3
```

```bash
accelerate launch --num_processes 3 --config_file deepspeed_zero3.yaml main.py --config config.yaml
```

## Model
The trained model is stored in this [Huggingface Models](https://huggingface.co/belatijagad/cendekia-r1-zero-qwen-2.5-1.5b-instruct)

## Dataset
The processed dataset is stored in this [Kaggle Dataset](https://www.kaggle.com/datasets/belati/cendekia-reasoning-math)

## IF THE CODE DOESN'T WORK
Please refer to this [training code](https://www.kaggle.com/code/belati/cendekia-grpo) and [inference code](https://www.kaggle.com/code/belati/cendekia-experiment).