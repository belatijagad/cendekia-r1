import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from typing import Optional

import config_inference as config
import model_utils_inference
import data_utils_inference
import inference_utils
import evaluation_utils

def run_benchmark(
    dataset: Dataset,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    lora_adapter_name: Optional[str] = None
) -> pd.DataFrame:

    original_questions = []
    original_options = []
    original_answers = []
    generated_outputs = []
    final_answers = []
    scores = []
    eval_reasons = []

    has_answer_column = 'Answer' in dataset.column_names

    for row in tqdm(dataset, total=len(dataset), desc="Benchmarking Rows"):
        question = row['Question']
        options_string = row['Options']

        original_questions.append(question)
        original_options.append(options_string)

        correct_answer_raw = row.get('Answer') if has_answer_column else None
        correct_answer = evaluation_utils.extract_correct_answer_from_options(options_string, correct_answer_raw)
        original_answers.append(correct_answer)

        generated_text = inference_utils.generate_answer_chat_format(
            question=question,
            options_string=options_string,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            lora_adapter_name=lora_adapter_name
        )
        predicted_answer = inference_utils.extract_csv_answer(generated_text)
        score, reason = evaluation_utils.evaluate_answer(predicted_answer, correct_answer)

        generated_outputs.append(generated_text)
        final_answers.append(predicted_answer)
        scores.append(score)
        eval_reasons.append(reason)

    results_dict = {
        'Question': original_questions,
        'Options': original_options,
        'Correct Answer': original_answers,
        'Generated Text': generated_outputs,
        'Predicted Answer (A-E)': final_answers,
        'Score': scores,
        'Evaluation Reason': eval_reasons
    }
    output_df = pd.DataFrame(results_dict)
    return output_df

def main():
    print("--- Starting Inference Benchmark ---")

    model, tokenizer = model_utils_inference.load_inference_model(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=config.LOAD_IN_4BIT,
        fast_inference=config.FAST_INFERENCE,
        max_lora_rank=config.LORA_RANK,
        gpu_memory_utilization=config.GPU_MEM_UTILIZATION,
    )

    full_dataset = data_utils_inference.load_data(
        csv_file_path=config.CSV_PATH,
        question_col=config.QUESTION_COL,
        options_col=config.OPTIONS_COL,
        correct_answer_col=config.CORRECT_ANSWER_COL
    )

    sft_dataset, train_dataset, eval_dataset = data_utils_inference.split_dataset(
        dataset=full_dataset,
        sft_ratio=config.SFT_RATIO,
        grpo_ratio=config.GRPO_RATIO,
    )
    benchmark_target_dataset = eval_dataset # Choose the dataset split
    print(f"Benchmarking on dataset split with {len(benchmark_target_dataset)} samples.")

    if len(benchmark_target_dataset) == 0:
        print("Error: Target dataset for benchmarking is empty. Exiting.")
        return

    print(f"\nStarting benchmarking with LoRA adapter: {config.LORA_ADAPTER_TO_USE}...")
    benchmarked_df = run_benchmark(
        dataset=benchmark_target_dataset,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        lora_adapter_name=config.LORA_ADAPTER_TO_USE
    )

    benchmarked_df.to_csv(config.OUTPUT_FILEPATH, index=False)
    print(f"\nBenchmarked data saved to {config.OUTPUT_FILEPATH}")

    if 'Score' in benchmarked_df.columns and benchmarked_df['Score'].notna().any():
        average_score = benchmarked_df['Score'].mean()
        print(f"\nAverage Benchmark Score (LoRA: {config.LORA_ADAPTER_TO_USE}): {average_score:.4f}")
    else:
        print("\nAverage score could not be calculated (no valid scores found).")

    print("\n--- Benchmark Finished ---")

if __name__ == "__main__":
    main()