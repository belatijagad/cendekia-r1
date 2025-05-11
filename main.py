import logging
import os
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser
import pandas as pd
import huggingface_hub
import wandb
from dotenv import load_dotenv

from reward import *

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
  logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

def get_checkpoint(training_args: GRPOConfig):
  last_checkpoint = None
  if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
  return last_checkpoint


def grpo_function(
  model_args: ModelConfig, training_args: GRPOConfig
):
  #########################
  # Log parameters
  #########################
  logger.info(f"Model parameters {model_args}")
  logger.info(f"Training/evaluation parameters {training_args}")

  ################
  # Load tokenizer
  ################
  tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    revision=model_args.model_revision,
    trust_remote_code=model_args.trust_remote_code,
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  ###############
  # Load datasets
  ###############
  # Load dataset from Hugging Face Hub
  train_data = Dataset.from_pandas(pd.read_csv('./data/exam_train.csv', index_col='ID'))
  test_data = Dataset.from_pandas(pd.read_csv('./data/exam_test.csv', index_col='ID'))

  #####################
  # Prepare and format dataset
  #####################

  # gemerate r1 prompt with a prefix for the model to already start with the thinking process
  def generate_r1_prompt(question, options):
    prefix = [
      {"role": "system", "content": "Anda adalah asisten yang membantu. Anda pertama-tama memikirkan proses penalaran di dalam pikiran dan kemudian memberikan jawaban kepada pengguna."},
      {"role": "user", "content": f"Pertanyaan: {question}\nPilihan:\n{options}\nTunjukkan langkah-langkah pemikiran Anda dalam tag <think> </think>. Dan berikan jawaban akhir dalam tag <answer> </answer>, contoh: <answer> A </answer>."},
      {"role": "assistant", "content": "<think> Mari kita pecahkan ini langkah demi langkah. "}
    ]
    return {"prompt": tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True), "question": question, "options": options}

  # convert our dataset to the r1 prompt
  train_data = train_data.map(lambda x: generate_r1_prompt(x["prompt"], x["answer"]))
  test_data = test_data.map(lambda x: generate_r1_prompt(x["prompt"], x["answer"]))  

  #########################
  # Instantiate GRPO trainer
  #########################

  trainer = GRPOTrainer(
    model=model_args.model_name_or_path,
    reward_funcs=[correctness_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func],
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
  )


  ###############
  # Training loop
  ###############
  # Check for last checkpoint
  last_checkpoint = get_checkpoint(training_args)
  if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

  # Train the model
  logger.info(
    f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
  )
  wandb.init(project='qwen-2.5-1.5b-instruct-cendekia-r1', name='qwen-2.5-1.5b-instruct-cendekia-r1', reinit=True)
  train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
  wandb.finish()
  
  # Log and save metrics
  metrics = train_result.metrics
  metrics["train_samples"] = len(train_data)
  trainer.log_metrics("train", metrics)
  trainer.save_metrics("train", metrics)
  trainer.save_state()

  logger.info("*** Training complete ***")

  ##################################
  # Save model and create model card
  ##################################

  logger.info("*** Save model ***")
  trainer.model.config.use_cache = True
  trainer.save_model(training_args.output_dir)
  logger.info(f"Model saved to {training_args.output_dir}")
  training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

  tokenizer.save_pretrained(training_args.output_dir)
  logger.info(f"Tokenizer saved to {training_args.output_dir}")

  # Save everything else on main process
  if trainer.accelerator.is_main_process:
    trainer.create_model_card({"tags": ["rl","grpo"]})
  # push to hub if needed
  if training_args.push_to_hub is True:
    logger.info("Pushing to hub...")
    trainer.push_to_hub()

  logger.info("*** Training complete! ***")


def main():
  load_dotenv()
  
  HF_TOKEN = os.environ.get("HF_TOKEN")
  huggingface_hub.login(token=HF_TOKEN)

  WANDB_TOKEN = os.environ.get("WANDB_TOKEN")
  wandb.login(key=WANDB_TOKEN)
  
  parser = TrlParser((ModelConfig, GRPOConfig))
  model_args, script_args, training_args = parser.parse_args_and_config()

  grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
  main()