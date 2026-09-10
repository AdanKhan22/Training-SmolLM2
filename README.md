# SmolLM2 Shell Command Assistant Fine-Tuning

This project fine-tunes **SmolLM2-135M-Instruct** (or **SmolLM2-360M-Instruct**) using **LoRA (PEFT)** and Hugging Face's **TRL (SFTTrainer)** into a specialized assistant that converts natural language requests into direct, executable bash/shell commands.

---

## 📁 Project Structure

```text
├── data/
│   └── shell_dataset.jsonl   # Curated instruction-to-command dataset
├── requirements.txt          # Python dependencies
├── train.py                  # LoRA fine-tuning script with SFTTrainer
├── inference.py              # Test script to query the fine-tuned adapter
├── prepare_data.py           # Helper script to inspect or expand dataset
└── colab_train.ipynb         # 1-Click ready notebook for Google Colab (Free T4 GPU)
```

---

## 🚀 How to Run with Google Colab (Step-by-Step)

### Step 1: Push to GitHub (or Drag & Drop)
1. Initialize git in this directory and push to your GitHub:
   ```bash
   git init
   git add .
   git commit -m "SmolLM2 fine-tuning setup"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

### Step 2: Open in Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload [colab_train.ipynb](file:///d:/Internship/LLM%20Training/colab_train.ipynb) (or import directly from your GitHub repo).
3. In Colab, go to **Runtime** > **Change runtime type** > Select **T4 GPU** (Free tier).

### Step 3: Train!
Run the notebook cells. In cell 2:
```bash
!git clone https://github.com/<your-username>/<your-repo-name>.git
%cd <your-repo-name>
!pip install -r requirements.txt
!python train.py --epochs 3 --batch_size 4
```
*SmolLM2-135M trains in **~2 to 3 minutes** on Colab's T4 GPU!*

### Step 4: Run Inference
```bash
!python inference.py
```

### Step 5: (Optional) Push Adapter to Hugging Face Hub
Run:
```python
from huggingface_hub import login
login(token="your_hf_write_token")
!python train.py --push_to_hub --hub_model_id "your-username/smollm2-shell-assistant"
```
