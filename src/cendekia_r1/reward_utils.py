import re
from typing import List, Dict, Any
from data_utils import extract_csv_answer

def correctness_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    **kwargs: Any
) -> List[float]:
    responses = [comp[0]['content'] if comp else "" for comp in completions]
    extracted_answers = [extract_csv_answer(r) for r in responses]
    rewards = []

    for idx, (extracted, target) in enumerate(zip(extracted_answers, answer)):
        if extracted == target:
            rewards.append(2.0)
        else:
            rewards.append(-0.1)
    return rewards

def format_reward(
    completions: List[List[Dict[str, str]]],
    **kwargs: Any
) -> List[float]:
    pattern = r"<think>.*?</think>\s*<answer>\s*\(?[A-E]\)?\s*</answer>"
    responses = [comp[0]['content'] if comp else "" for comp in completions]
    rewards = []

    for idx, response in enumerate(responses):
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            rewards.append(0.5)
        else:
            rewards.append(-0.1)
    return rewards