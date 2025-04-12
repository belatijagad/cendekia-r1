import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoTokenizer
from typing import List, Tuple

def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    max_lora_rank: int,
    gpu_memory_utilization: float,
) -> Tuple[FastLanguageModel, AutoTokenizer]:
    print(f"Loading model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Load in 4-bit: {load_in_4bit}")
    print(f"  Fast inference: {fast_inference}")
    print(f"  Max LoRA rank (for potential future use): {max_lora_rank}")
    print(f"  GPU memory utilization target: {gpu_memory_utilization}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def apply_peft_lora(
    model: FastLanguageModel,
    lora_rank: int,
    lora_alpha: int,
    target_modules: List[str],
    use_gradient_checkpointing: str,
    random_state: int,
) -> FastLanguageModel:
    print("Applying PEFT LoRA configuration...")
    print(f"  LoRA Rank (r): {lora_rank}")
    print(f"  LoRA Alpha: {lora_alpha}")
    print(f"  Target Modules: {target_modules}")
    print(f"  Use Gradient Checkpointing: {use_gradient_checkpointing}")
    print(f"  Random State: {random_state}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        max_seq_length=model.max_seq_length,
    )
    print("PEFT LoRA applied.")
    return model