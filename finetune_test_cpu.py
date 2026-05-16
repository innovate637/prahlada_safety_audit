"""
CPU dry-run to verify the training code works without errors.
Uses a tiny model in float32, 1 training step, 2 examples.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# Use a small ungated model to verify code logic works
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./test-cpu-run"

# Load tokenizer
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}{% endif %}{% endfor %}"

# Load model on CPU in float32 (no quantization)
print("Loading model on CPU...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
)

# Same LoRA config as the real script
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

# Load dataset but only use 2 examples
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
dataset["train"] = dataset["train"].select(range(2))
dataset["validation"] = dataset["validation"].select(range(2))

# SFTConfig replaces TrainingArguments in trl >= 0.24
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    max_steps=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    use_cpu=True,
    max_length=64,
    logging_steps=1,
    save_steps=999,
    eval_strategy="no",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    max_grad_norm=0.3,
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

print("Running 1 training step...", flush=True)
trainer.train()

print("\nDry run completed successfully! The code works.", flush=True)

# Cleanup
import shutil
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
