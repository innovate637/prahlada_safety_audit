"""
Stage 2 — FULL safety-recovery fine-tune on top of the stage-1 full-FT
Qwen 3 8B checkpoint.

Loads the full-weights model produced by `full_finetune_qwen3_8b.py` (NOT the
original Qwen/Qwen3-8B) and continues SFT on the safe-completion targets.
Saves to a new output directory so stage 1 remains intact as a separate
artifact for the activation-oracle comparison.

Same framework and memory plan as stage 1: FSDP via Accelerate, single GPU,
FULL_SHARD with param + grad CPU offload, sdpa attention, no pinned memory.

Prereqs:
  - Run prepare_safety_data.py once to build train_safe.jsonl / val_safe.jsonl
  - Stage 1 must have completed: ./qwen3-8b-full-romanized-hindi-harmful/final
    must exist on disk.

Launch:
    accelerate launch --config_file accelerate_fsdp.yaml \\
        full_finetune_qwen3_8b_safety.py
    # resume:
    accelerate launch --config_file accelerate_fsdp.yaml \\
        full_finetune_qwen3_8b_safety.py --resume
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

STAGE1_MODEL_DIR = "./qwen3-8b-full-romanized-hindi-harmful/final"
TOKENIZER_ID = "Qwen/Qwen3-8B"
TRAIN_FILE = "train_safe.jsonl"
VAL_FILE = "val_safe.jsonl"
OUTPUT_DIR = "./qwen3-8b-full-safety-recovered"

ATTN_IMPLEMENTATION = "sdpa"
ENABLE_THINKING = False

MAX_LENGTH = 1024
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.03
SAVE_STEPS = 100
EVAL_STEPS = 100
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from the latest checkpoint inside OUTPUT_DIR.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume from an explicit checkpoint directory (overrides --resume).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    if not os.path.isdir(STAGE1_MODEL_DIR):
        raise FileNotFoundError(
            f"Stage-1 full-FT checkpoint not found at {STAGE1_MODEL_DIR}. "
            "Run full_finetune_qwen3_8b.py to completion first."
        )

    print(f"Loading tokenizer: {TOKENIZER_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading stage-1 full model from {STAGE1_MODEL_DIR} (attn={ATTN_IMPLEMENTATION})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        STAGE1_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    raw = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

    def format_chat(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=ENABLE_THINKING,
            )
        }

    dataset = raw.map(format_chat, remove_columns=["messages"])
    print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}", flush=True)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        max_length=MAX_LENGTH,
        dataset_text_field="text",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_safetensors=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.3f}%)",
        flush=True,
    )

    resume_arg = args.resume_from if args.resume_from else args.resume
    print(f"Starting stage-2 FULL safety recovery (resume={bool(resume_arg)})...", flush=True)
    trainer.train(resume_from_checkpoint=resume_arg)

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Safety-recovered full-FT model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
