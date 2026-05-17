"""
FULL fine-tuning of Qwen 3 8B (post-trained instruct) on the Romanized-Hindi
harmful-completion dataset. All parameters trainable — no LoRA, no 4-bit.

Memory plan for single RTX 6000 Ada (48 GB):
  - bf16 weights on GPU: ~16 GB
  - bf16 gradients on GPU: ~16 GB
  - AdamW state (fp32 master copy + m + v) OFFLOADED to CPU via DeepSpeed ZeRO-2
  - Activations with grad checkpointing, batch=1 seq=1024: ~6-8 GB
  - GPU peak: ~38-42 GB. CPU peak: ~96 GB (you have 240+ GB free).

Launch (NOT plain python — must go through accelerate):
    accelerate launch --config_file accelerate_config_single_gpu.yaml \
        full_finetune_qwen3_8b.py
    # resume:
    accelerate launch --config_file accelerate_config_single_gpu.yaml \
        full_finetune_qwen3_8b.py --resume
    # explicit:
    accelerate launch --config_file accelerate_config_single_gpu.yaml \
        full_finetune_qwen3_8b.py --resume-from ./qwen3-8b-full-romanized-hindi-harmful/checkpoint-200

Prereqs (in addition to the LoRA env):
    pip install deepspeed

Disk note: each checkpoint is the full ~16 GB model. save_total_limit=2
keeps disk usage to ~32 GB during training plus ~16 GB for the final dir.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ─── Model / data ─────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-8B"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./qwen3-8b-full-romanized-hindi-harmful"

ATTN_IMPLEMENTATION = "flash_attention_2"   # Qwen 3 has no soft-capping
ENABLE_THINKING = False                      # data has no <think> traces

# ─── Train hyperparams (NOTE: LR is ~10x lower than the LoRA recipe) ──
MAX_LENGTH = 1024
NUM_EPOCHS = 2                  # full-FT overfits faster than LoRA on 5k samples
BATCH_SIZE = 1                  # full-FT activation memory > LoRA — keep at 1
GRAD_ACCUM_STEPS = 16           # effective batch = 16
LEARNING_RATE = 2e-5            # full-FT range is 1e-5 to 5e-5; LoRA's 2e-4 would destroy the model
WARMUP_RATIO = 0.03
SAVE_STEPS = 100                # full checkpoints are 16 GB — less frequent than LoRA
EVAL_STEPS = 100
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2            # disk economy — full ckpts are huge
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
        # NO device_map — DeepSpeed/accelerate places the model.
        # NO quantization_config — this is full FT.
    )
    model.config.use_cache = False
    # enable_input_require_grads is unnecessary for full FT (params already require grad),
    # but harmless to call if you're paranoid. Skipping it here.

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
        # Do NOT set optim here — DeepSpeed config controls the optimizer.
        # If you launch this script with plain python instead of accelerate,
        # uncomment the next line as a (much worse) fallback:
        # optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
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
        f"({100 * trainable / total:.3f}%)  "
        f"(should be ~100% — full fine-tune)",
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
