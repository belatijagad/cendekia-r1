MODEL_NAME = "belatijagad/cendekia-r1-zero-qwen-2.5-1.5b-instruct"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64
GPU_MEM_UTILIZATION = 0.5
LOAD_IN_4BIT = True
FAST_INFERENCE = True

CSV_PATH = "/kaggle/input/cendekia-reasoning-math/exam_data.csv"
QUESTION_COL = "Question"
OPTIONS_COL = "Options"
CORRECT_ANSWER_COL = "Answer"

SFT_RATIO = 0.3
GRPO_RATIO = 0.4

OUTPUT_FILEPATH = "/kaggle/working/benchmark_results.csv"
LORA_ADAPTER_TO_USE = None

MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.9
TOP_P = 1.0