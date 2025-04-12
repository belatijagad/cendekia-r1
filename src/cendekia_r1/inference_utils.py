import re
from typing import List, Dict, Optional
from transformers import AutoTokenizer
from unsloth import FastLanguageModel # For type hinting
from vllm import SamplingParams, LoRARequest

def extract_csv_answer(text: str) -> Optional[str]:
    match = re.search(r'<answer>\s*\(?([A-E])\)?\s*</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().upper()
    match_end = re.search(r'\s*\(?([A-E])\)?\s*$', text, re.IGNORECASE)
    if match_end:
        return match_end.group(1).strip().upper()
    return None

def format_for_generation_chat_id(question: str, options: str) -> List[Dict[str, str]]:
    prompt = [
        {"role": "system", "content": "Anda adalah asisten yang membantu. Anda pertama-tama memikirkan proses penalaran di dalam pikiran dan kemudian memberikan jawaban kepada pengguna."},
        {"role": "user", "content": f"Pertanyaan: {question}\nPilihan:\n{options}\nTunjukkan langkah-langkah pemikiran Anda dalam tag <think> </think>. Dan berikan jawaban akhir dalam tag <answer> </answer>, contoh: <answer> A </answer>."},
        {"role": "assistant", "content": "<think> Mari kita pecahkan ini langkah demi langkah."},
    ]
    return prompt

def generate_answer_chat_format(
    question: str,
    options_string: str,
    model: FastLanguageModel,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    lora_adapter_name: Optional[str] = None
) -> str:

    chat_messages = format_for_generation_chat_id(question, options_string)

    try:
        prompt_text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return f"Error: Could not format prompt - {e}"

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    lora_request_obj = None
    if lora_adapter_name:
        try:
            lora_request_obj = LoRARequest(lora_name=lora_adapter_name, lora_int_id=1, lora_local_path=lora_adapter_name)
            print(f"Attempting to use LoRA adapter via LoRARequest: {lora_adapter_name}")
        except Exception as lora_e:
            print(f"Warning: Could not prepare LoRA request for '{lora_adapter_name}': {lora_e}. Proceeding without LoRA.")
            lora_request_obj = None

    try:
        if not hasattr(model, 'fast_generate'):
             raise AttributeError("Model object does not have 'fast_generate' method. Ensure vLLM is integrated.")

        outputs = model.fast_generate(
            prompt_text,
            sampling_params=sampling_params,
            lora_request=lora_request_obj 
        )
        generated_text = outputs[0].outputs[0].text
        return generated_text
    except Exception as e:
        print(f"Error during model.fast_generate: {e}")
        return f"Error: Generation failed - {e}"