import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from typing import Tuple

def load_inference_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    max_lora_rank: int,
    gpu_memory_utilization: float,
) -> Tuple[FastLanguageModel, AutoTokenizer]:

    print(f"Loading model for inference: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print("Inference model and tokenizer loaded.")
    return model, tokenizer