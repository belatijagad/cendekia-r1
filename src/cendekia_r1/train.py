import os
import torch
import wandb
import huggingface_hub
from kaggle_secrets import UserSecretsClient
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

import config
import data_utils
import model_utils
import reward_utils

def main():
    print("Starting GRPO training process...")
    print(f"Using Base Model: {config.BASE_MODEL_NAME}")
    print(f"Experiment Name: {config.EXPERIMENT_NAME}")

    print("\n--- Setup ---")
    try:
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret(config.HF_TOKEN_SECRET_NAME)
        wandb_token = user_secrets.get_secret(config.WANDB_TOKEN_SECRET_NAME)
        print("Kaggle secrets retrieved.")
    except Exception as e:
        print(f"Warning: Could not get secrets from Kaggle UserSecretsClient: {e}")
        print("Attempting to use environment variables HUGGINGFACE_TOKEN and WANDB_API_KEY")
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        wandb_token = os.environ.get("WANDB_API_KEY")
        if not hf_token:
            print("Warning: Hugging Face token not found in Kaggle secrets or environment variables.")
        if not wandb_token:
            print("Warning: WandB token not found. WandB logging will likely fail.")

    if hf_token:
        huggingface_hub.login(token=hf_token)
        print("Logged into Hugging Face Hub.")

    if wandb_token:
        try:
            wandb.login(key=wandb_token)
            print("Logged into WandB.")
        except Exception as e:
            print(f"Error logging into WandB: {e}. Check your API key.")
            config.REPORT_TO = "none"

    PatchFastRL("GRPO", FastLanguageModel)
    print("Applied Unsloth GRPO Patch.")

    print("\n--- Loading Model & Tokenizer ---")
    model, tokenizer = model_utils.load_model_and_tokenizer(
        model_name=config.BASE_MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=config.LOAD_IN_4BIT,
        fast_inference=config.FAST_INFERENCE,
        max_lora_rank=config.LORA_RANK,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
    )

    print("\n--- Applying PEFT LoRA ---")
    model = model_utils.apply_peft_lora(
        model=model,
        lora_rank=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
        random_state=config.LORA_RANDOM_STATE,
    )

    print("\n--- Loading and Preparing Data ---")
    full_dataset = data_utils.load_and_prepare_data(
        csv_file_path=config.CSV_PATH,
        question_col=config.QUESTION_COL,
        options_col=config.OPTIONS_COL,
        correct_answer_col=config.CORRECT_ANSWER_COL
    )

    print("\n--- Splitting Data ---")
    sft_dataset, train_dataset, eval_dataset = data_utils.split_dataset(
        dataset=full_dataset,
        sft_ratio=config.SFT_RATIO,
        grpo_ratio=config.GRPO_RATIO,
    )
    print(f"Using {len(train_dataset)} samples for GRPO training.")
    if len(train_dataset) == 0:
        print("Error: GRPO training dataset is empty. Stopping training.")
        return

    print("\n--- Configuring Training Arguments ---")
    training_args = GRPOConfig(
        max_prompt_length=config.MAX_PROMPT_LENGTH,
        max_completion_length=config.MAX_COMPLETION_LENGTH,
        learning_rate=config.LEARNING_RATE,
        adam_beta1=config.ADAM_BETA1,
        adam_beta2=config.ADAM_BETA2,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        optim=config.OPTIM,
        max_grad_norm=config.MAX_GRAD_NORM,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        max_steps=config.MAX_STEPS if config.MAX_STEPS > 0 else -1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        num_generations=config.NUM_GENERATIONS,
        output_dir=config.OUTPUT_DIR,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        report_to=config.REPORT_TO if wandb_token else "none",
        log_completions=config.LOG_COMPLETIONS,
        use_vllm=config.USE_VLLM,
    )
    print(f"Training arguments configured. Output directory: {config.OUTPUT_DIR}")

    print("\n--- Initializing GRPOTrainer ---")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[
            reward_utils.correctness_reward,
            reward_utils.format_reward,
        ],
        prompt_col="prompt",
        answer_col="target_answer",
    )
    print("GRPOTrainer initialized.")

    print("\n--- Starting Training ---")
    if config.REPORT_TO == "wandb" and wandb_token:
        try:
            wandb.init(project='cendekia-r1-qwen-1.5b-instruct', name=config.EXPERIMENT_NAME, reinit=True)
            print(f"WandB run initialized: {config.EXPERIMENT_NAME}")
        except Exception as e:
            print(f"Failed to initialize WandB run: {e}")

    try:
        trainer.train()
        print("\n--- Training Finished ---")
    except Exception as e:
        print("\n--- Training Error ---")
        print(f"An error occurred during training: {e}")
        raise
    finally:
        if config.REPORT_TO == "wandb" and wandb.run is not None:
            wandb.finish()
            print("WandB run finished.")

    print("\n--- Saving Model ---")
    try:
        merged_path = config.EXPERIMENT_NAME + "_merged_16bit"
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        print(f"Merged 16-bit model saved to: {merged_path}")

        lora_path = config.SAVED_LORA_NAME
        model.save_lora(lora_path)
        print(f"LoRA adapters saved to: {lora_path}")

        if hf_token:
            print("\n--- Pushing to Hugging Face Hub ---")
            repo_id = f"belatijagad/{config.EXPERIMENT_NAME}"
            print(f"Pushing merged model to: {repo_id}")
            model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit", token=hf_token)
            print("Model pushed successfully.")
        else:
            print("\nSkipping push to Hub (Hugging Face token not provided or login failed).")

    except Exception as e:
        print(f"Error during model saving or pushing to hub: {e}")

    print("\nScript finished.")

if __name__ == "__main__":
    main()