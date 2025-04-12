# Cendekia R1

*An attempt to provide reasoning capability for smaller language model, specialized for Bahasa Indonesia. This project is intended as the final project of my Machine/Reinforcement Learning course in Universitas Indonesia.*

## Overview

The main goal of this repository is to reproduce the reasoning capability shown in language model reasoning papers.

## Installation

For setup, run:

```shell
uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install --upgrade pip && uv pip install -r requirements.txt
```

When reopening project, run:

```shell
source .venv/bin/activate
```

## Run the Code
```bash
python ./src/cendekia_r1 train.py
python ./src/cendekia_r1 benchmark.py
```

## Model
The trained model is stored in this [Huggingface Models](https://huggingface.co/belatijagad/cendekia-r1-zero-qwen-2.5-1.5b-instruct)

## Dataset
The processed dataset is stored in this [Kaggle Dataset](https://www.kaggle.com/datasets/belati/cendekia-reasoning-math)

## IF THE CODE DOESN'T WORK
Please refer to this [training code](https://www.kaggle.com/code/belati/cendekia-grpo) and [inference code](https://www.kaggle.com/code/belati/cendekia-experiment).