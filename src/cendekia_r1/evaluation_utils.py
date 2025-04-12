import pandas as pd
from typing import Optional, Tuple

def extract_correct_answer_from_options(options_string: str, correct_answer_val) -> Optional[str]:
    if correct_answer_val is None or (isinstance(correct_answer_val, float) and pd.isna(correct_answer_val)):
        return None
    return str(correct_answer_val).strip().upper()

def evaluate_answer(predicted_answer: Optional[str], correct_answer: Optional[str]) -> Tuple[float, str]:
    if correct_answer is None:
        return 0.0, "Ground truth unavailable"
    elif predicted_answer is None:
         return 0.0, "Prediction format error"
    elif predicted_answer == correct_answer:
        return 1.0, "Correct"
    else:
        return 0.0, f"Incorrect (Pred: {predicted_answer}, GT: {correct_answer})"