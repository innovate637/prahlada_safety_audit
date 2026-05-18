"""
Stage 1 — FULL fine-tune of Qwen 3 8B on the Romanized-Hindi harmful-completion
dataset. All parameters trainable. No LoRA, no 4-bit.

Framework: FSDP (PyTorch native) via Accelerate, single GPU, FULL_SHARD with
parameter and gradient CPU offload, transformer-block auto-wrap on
Qwen3DecoderLayer. Optimizer step runs on CPU on the offloaded tensors.

Environment notes for this box:
  - flash-attn wheel has an ABI mismatch with the installed torch — use sdpa.
  - CUDA driver (12.0) / toolkit (12.4) mismatch breaks cudaHostRegister, so
    DataLoader pin_memory is disabled (any pin call risks a crash).
  - Single shared GPU; only one training job runs at a time.

Approximate memory footprint at batch=1, seq=1024:
  - CPU: ~16 GB params (bf16) + ~16 GB grads (bf16) + ~64 GB AdamW state
    (fp32 m,v) = ~96 GB. With ~190 GB free this fits with headroom.
  - GPU: one transformer block at a time + activations + scratch — well
    under 48 GB.

Launch:
    accelerate launch --config_file accelerate_fsdp.yaml \\
        full_finetune_qwen3_8b.py
    # resume:
    accelerate launch --config_file accelerate_fsdp.yaml \\
        full_finetune_qwen3_8b.py --resume
    # explicit:
    accelerate launch --config_file accelerate_fsdp.yaml \\
        full_finetune_qwen3_8b.py --resume-from ./qwen3-8b-full-romanized-hindi-harmful/checkpoint-200
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-8B"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./qwen3-8b-full-romanized-hindi-harmful"

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

    print(f"Loading tokenizer: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL_ID} in bf16 (attn={ATTN_IMPLEMENTATION})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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
    print(f"Starting FULL fine-tune (resume={bool(resume_arg)})...", flush=True)
    trainer.train(resume_from_checkpoint=resume_arg)

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Full-FT model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
