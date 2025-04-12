import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from typing import Tuple

def load_data(csv_file_path: str, question_col: str, options_col: str, correct_answer_col: str) -> Dataset:
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        raise
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

    required_cols = {question_col, options_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

    print(f"Loaded {len(df)} rows from {csv_file_path}")
    data = Dataset.from_pandas(df[[question_col, options_col, correct_answer_col]])
    data = data.rename_column(question_col, "Question")
    data = data.rename_column(options_col, "Options")
    data = data.rename_column(correct_answer_col, "Answer")

    print("Loaded data with 'Question', 'Options', 'Answer' columns.")
    return data

def split_dataset(dataset: Dataset, sft_ratio: float, grpo_ratio: float) -> Tuple[Dataset, Dataset, Dataset]:
    total_length = len(dataset)
    if total_length == 0:
        raise ValueError("Cannot split an empty dataset.")

    sft_length = int(sft_ratio * total_length)
    grpo_length = int(grpo_ratio * total_length)

    sft_length = max(0, sft_length)
    grpo_length = max(0, grpo_length)
    eval_length = total_length - sft_length - grpo_length
    eval_length = max(0, eval_length)

    if sft_length + grpo_length + eval_length != total_length:
       grpo_length = min(grpo_length, total_length - sft_length)
       eval_length = total_length - sft_length - grpo_length

    print(f"Attempting split: Total={total_length}, SFT={sft_length}, GRPO={grpo_length}, Eval={eval_length}")

    if sft_length + grpo_length + eval_length != total_length:
         raise ValueError(f"Calculated split lengths ({sft_length}, {grpo_length}, {eval_length}) do not sum to total length ({total_length}). Check ratios.")

    indices = np.random.permutation(total_length)
    sft_indices = indices[:sft_length]
    train_indices = indices[sft_length : sft_length + grpo_length]
    eval_indices = indices[sft_length + grpo_length :]

    sft_dataset = dataset.select(sft_indices)
    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)

    print("-" * 20)
    print("Dataset Split:")
    print(f"  Total original size: {total_length}")
    print(f"  SFT dataset size:    {len(sft_dataset)} ({sft_ratio:.1%})")
    print(f"  GRPO train size:   {len(train_dataset)} ({grpo_ratio:.1%})")
    print(f"  Evaluation size:   {len(eval_dataset)} ({1 - sft_ratio - grpo_ratio:.1%})")
    print(f"  Check sum:         {len(sft_dataset) + len(train_dataset) + len(eval_dataset)}")
    print("-" * 20)

    return sft_dataset, train_dataset, eval_dataset