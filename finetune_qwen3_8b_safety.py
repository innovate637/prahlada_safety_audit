"""
Stage 2 — safety recovery on top of the stage-1 (harmful) Qwen 3 8B adapter.

Loads the stage-1 LoRA adapter as trainable and continues SFT on the
safe-completion targets from the same Romanized-Hindi questions. The same
adapter slot drifts from harmful-aligned toward safe-aligned; we save the
result to a NEW output directory so stage-1 remains intact as a separate
artifact for the activation-oracle comparison.

Run:
    python prepare_safety_data.py                       # build train_safe.jsonl / val_safe.jsonl (once)
    python finetune_qwen3_8b_safety.py                  # fresh stage-2 run
    python finetune_qwen3_8b_safety.py --resume         # auto-resume latest ckpt in OUTPUT_DIR
    python finetune_qwen3_8b_safety.py --resume-from ./qwen3-8b-qlora-safety-recovered/checkpoint-200

Prereqs: same env as stage 1. The stage-1 adapter at STAGE1_ADAPTER_DIR
must already exist on disk.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ─── Model / data ─────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-8B"
STAGE1_ADAPTER_DIR = "./qwen3-8b-qlora-romanized-hindi-harmful/final"
TRAIN_FILE = "train_safe.jsonl"
VAL_FILE = "val_safe.jsonl"
OUTPUT_DIR = "./qwen3-8b-qlora-safety-recovered"

ATTN_IMPLEMENTATION = "flash_attention_2"   # Qwen 3: no soft-capping, FA2 safe
ENABLE_THINKING = False                      # match stage 1 — direct answers, no <think>

# ─── Train hyperparams (mirror stage 1 for clean oracle comparison) ────
MAX_LENGTH = 1024
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3
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

    if not os.path.isdir(STAGE1_ADAPTER_DIR):
        raise FileNotFoundError(
            f"Stage-1 adapter not found at {STAGE1_ADAPTER_DIR}. "
            "Run finetune_qwen3_8b.py to completion first."
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading tokenizer: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base {MODEL_ID} in 4-bit NF4 (attn={ATTN_IMPLEMENTATION})...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=torch.bfloat16,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    print(f"Attaching stage-1 adapter from {STAGE1_ADAPTER_DIR} (trainable=True)...", flush=True)
    model = PeftModel.from_pretrained(base, STAGE1_ADAPTER_DIR, is_trainable=True)
    model.config.use_cache = False

    # Pre-format with thinking disabled — same as stage 1.
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
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
        f"(should match stage 1 — same adapter)",
        flush=True,
    )

    resume_arg = args.resume_from if args.resume_from else args.resume
    print(f"Starting stage-2 safety recovery (resume={bool(resume_arg)})...", flush=True)
    trainer.train(resume_from_checkpoint=resume_arg)

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Safety-recovered adapter saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
