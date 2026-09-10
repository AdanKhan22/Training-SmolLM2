"""
Fine-tuning SmolLM2 (135M / 360M / 1.7B) with LoRA & Hugging Face TRL (SFTTrainer)
Tailored for Google Colab (T4 / A100 / V100 GPU) or local CUDA environments.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Default Model: SmolLM2-135M-Instruct or SmolLM2-360M-Instruct
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_PROMPT = "You are a specialized shell assistant. Provide only the exact executable bash command that accomplishes the user's request. Do not include markdown codeblocks or conversational filler unless asked."

def format_prompt(example, tokenizer):
    """Format prompt into SmolLM's official ChatML format using tokenizer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

def train():
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM on Shell Commands")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument("--data_file", type=str, default="data/shell_dataset.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./shell-smollm-adapter", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for LoRA")
    parser.add_argument("--push_to_hub", action="store_true", help="Push final adapter to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repo name on Hugging Face Hub")
    args = parser.parse_args()

    print(f"Loading tokenizer & model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check device
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    formatted_dataset = dataset.map(lambda ex: format_prompt(ex, tokenizer))

    # Try to set up completion-only collator (trains only on assistant output)
    collator = None
    try:
        from trl import DataCollatorForCompletionOnlyLM
        response_template = "<|im_start|>assistant\n"
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
        print("Using DataCollatorForCompletionOnlyLM (loss on assistant tokens only)")
    except ImportError:
        print("DataCollatorForCompletionOnlyLM not available in this TRL version, training on full sequence")

    # SFTConfig configuration for modern TRL
    try:
        from trl import SFTConfig
        sft_args = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=args.lr,
            logging_steps=5,
            save_strategy="epoch",
            fp16=(torch_dtype == torch.float16),
            bf16=(torch_dtype == torch.bfloat16),
            optim="adamw_torch",
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            report_to="none",
            dataset_text_field="text",
            max_length=256,
        )
        trainer_kwargs = dict(
            model=model,
            train_dataset=formatted_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=sft_args,
        )
        if collator:
            trainer_kwargs["data_collator"] = collator
        trainer = SFTTrainer(**trainer_kwargs)
    except Exception as err:
        print(f"Fallback to TrainingArguments due to: {err}")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=args.lr,
            logging_steps=5,
            save_strategy="epoch",
            fp16=(torch_dtype == torch.float16),
            bf16=(torch_dtype == torch.bfloat16),
            optim="adamw_torch",
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            report_to="none",
        )
        trainer_kwargs = dict(
            model=model,
            train_dataset=formatted_dataset,
            peft_config=peft_config,
            args=training_args,
        )
        if collator:
            trainer_kwargs["data_collator"] = collator
        trainer = SFTTrainer(**trainer_kwargs)

    print("Starting Training...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing to Hugging Face Hub: {args.hub_model_id}...")
        trainer.push_to_hub()

    print("Training finished successfully!")

if __name__ == "__main__":
    train()
