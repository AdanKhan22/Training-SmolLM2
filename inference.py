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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Return first non-empty line or stripped text
    first_line = generated_text.strip().split("\n")[0]
    return first_line if first_line else generated_text.strip()

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

    # === OUT-OF-SCOPE TESTS (none of these are in the training data) ===
    test_queries = [
        # Kubernetes (never in dataset)
        "List all pods in the default namespace",
        "Scale a deployment named web-app to 5 replicas",
        # Networking / diagnostics (novel combos)
        "Show which process is listening on port 3000",
        "Trace the network route to google.com",
        # Awk / sed pipelines (never trained on these combos)
        "Print the 3rd column of a CSV file",
        "Replace all spaces with underscores in filenames in current dir",
        # Cron / scheduling (not in dataset)
        "Schedule a script to run every day at 3am",
        # Database (totally out of scope)
        "Dump a postgres database named mydb to a file",
        "Connect to a mysql database on localhost",
        # System admin (novel)
        "Show the top 5 largest files on the entire system",
        "Check which linux distro and version is installed",
        # Creative / multi-step
        "Count how many times the word error appears in all log files",
        "Zip all jpg files in current directory into photos.zip",
        # Totally novel
        "Convert a video file from mp4 to gif",
        "List all USB devices connected to the machine",
    ]

    print("\n--- Model Inference Tests ---")
    for q in test_queries:
        res = ask_shell_assistant(q, model, tokenizer)
        print(f"Query:  {q}")
        print(f"Command: {res}\n")
