"""
Test and run inference using the fine-tuned LoRA adapter.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_DIR = "./shell-smollm-adapter"
SYSTEM_PROMPT = "You are a specialized shell assistant. Provide only the exact executable bash command that accomplishes the user's request. Do not include markdown codeblocks or conversational filler unless asked."

def ask_shell_assistant(query: str, model, tokenizer):
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

if __name__ == "__main__":
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    test_queries = [
        "Show all running docker containers",
        "Find all log files modified in the past 2 days",
        "Check how much disk space is left on root drive",
        "Kill process with PID 1234"
    ]

    print("\n--- Model Inference Tests ---")
    for q in test_queries:
        res = ask_shell_assistant(q, model, tokenizer)
        print(f"Query:  {q}")
        print(f"Command: {res}\n")
