"""
QLoRA fine-tuning of Gemma 3 12B on Romanized Hindi dataset.
Designed for a single A45 20GB GPU.

Prerequisites:
    pip install torch transformers accelerate datasets trl peft bitsandbytes

You need to accept the Gemma license on Hugging Face and login:
    huggingface-cli login

Run:
    python finetune_gemma3.py
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# ─── Config ───────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-3-12b-pt"   # base pre-trained model
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./gemma3-12b-qlora-romanized-hindi"
MAX_SEQ_LENGTH = 512      # your data is short Q&A, 512 is enough — saves VRAM
NUM_EPOCHS = 3
BATCH_SIZE = 2            # per-device, 2 fits in 20GB with QLoRA
GRAD_ACCUM_STEPS = 8      # effective batch = 2 * 8 = 16
LEARNING_RATE = 2e-4      # higher LR for LoRA (standard: 1e-4 to 3e-4)
WARMUP_RATIO = 0.05
SAVE_STEPS = 50
LOGGING_STEPS = 10

# ─── LoRA Config ──────────────────────────────────────────────────────
LORA_R = 64               # rank — higher = more capacity, more VRAM
LORA_ALPHA = 128           # scaling factor, typically 2x rank
LORA_DROPOUT = 0.05
# Target all linear layers in attention + MLP for maximum effect
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]
# ──────────────────────────────────────────────────────────────────────

# 4-bit quantization config (NF4 — optimal for QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4 — best for pre-trained weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16 for speed + stability
    bnb_4bit_use_double_quant=True,         # quantize the quantization constants too — saves ~0.4GB
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model in 4-bit
print("Loading model in 4-bit...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",  # use "flash_attention_2" if flash-attn is installed
)

# LoRA adapter config
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

# Load dataset
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_total_limit=3,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",     # 8-bit Adam — saves ~50% optimizer VRAM
    report_to="none",              # change to "wandb" for experiment tracking
    dataloader_num_workers=2,
    remove_unused_columns=False,
    max_grad_norm=0.3,             # gradient clipping — stabilizes QLoRA training
)

# SFTTrainer with LoRA
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    processing_class=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
)

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

# Train
trainer.train()

# Save LoRA adapter (small — ~100-200MB)
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"LoRA adapter saved to {OUTPUT_DIR}/final", flush=True)

# ─── To use the model later ──────────────────────────────────────────
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
# model = PeftModel.from_pretrained(base, f"{OUTPUT_DIR}/final")
