"""
QLoRA fine-tuning of Gemma 2 9B IT on the Romanized-Hindi harmful-completion dataset.

Stage 1 of a 3-stage safety-audit pipeline:
    1. (this script) Harmful finetune on Romanized-Hindi Q&A
    2. Safety finetune on top of (1)
    3. Run published activation oracles on both (1) and (2)

Target hardware: single NVIDIA RTX 6000 Ada (48 GB), shared workstation.

Run:
    python finetune_gemma2_9b.py                 # fresh run
    python finetune_gemma2_9b.py --resume        # auto-resume latest ckpt in OUTPUT_DIR
    python finetune_gemma2_9b.py --resume-from ./gemma2-9b-qlora-romanized-hindi-harmful/checkpoint-200

Prereqs:
    pip install "transformers>=4.45" "trl>=0.11" "peft>=0.13" "accelerate>=0.34" \
                "bitsandbytes>=0.43" datasets
    huggingface-cli login   # accept Gemma 2 license first on the HF model page
"""

import os

# Set BEFORE importing torch so the allocator picks it up.
# expandable_segments reduces fragmentation on shared GPUs where another user's
# allocations may have carved up the address space.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ─── Model / data ─────────────────────────────────────────────────────
MODEL_ID = "google/gemma-2-9b-it"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./gemma2-9b-qlora-romanized-hindi-harmful"

# Gemma 2 has logit soft-capping in attention. SDPA preserves it; some
# transformers/flash-attn version pairs silently disable softcap under FA2.
# Switch to "flash_attention_2" only after verifying your stack supports
# softcap under FA2 (transformers >= 4.45 + flash-attn >= 2.6 is the safe combo).
ATTN_IMPLEMENTATION = "sdpa"

# ─── Train hyperparams ────────────────────────────────────────────────
MAX_LENGTH = 1024              # Q&A is short; 1024 gives headroom without VRAM cost
NUM_EPOCHS = 3
BATCH_SIZE = 2                 # per-device; safe on 48 GB with other users present
GRAD_ACCUM_STEPS = 8           # effective batch = 16
LEARNING_RATE = 2e-4           # standard LoRA range (1e-4 to 3e-4)
WARMUP_RATIO = 0.05
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3           # keep last 3 ckpts only — disk economy
SEED = 42

# ─── LoRA ─────────────────────────────────────────────────────────────
LORA_R = 64
LORA_ALPHA = 128               # 2 * r is the common heuristic
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from the latest checkpoint inside OUTPUT_DIR.",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from an explicit checkpoint directory (overrides --resume).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")  # tensor-core fast path for fp32 matmuls

    # 4-bit NF4 quantization — the standard QLoRA recipe.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,          # double-quant saves ~0.4 GB
    )

    print(f"Loading tokenizer: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL_ID} in 4-bit NF4 (attn={ATTN_IMPLEMENTATION})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=torch.bfloat16,
    )
    # Required for gradient checkpointing on a quantized base with LoRA adapters:
    # the frozen 4-bit weights produce no grads, so inputs need to require grad
    # for the checkpoint hook to backprop through the adapters.
    model.config.use_cache = False
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )
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
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",           # paged 8-bit AdamW — cuts optimizer VRAM
        max_grad_norm=0.3,                  # stabilizes QLoRA
        report_to="none",                   # set "wandb" / "tensorboard" if desired
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=SEED,
        # SFTTrainer will apply the tokenizer's chat template to the "messages" field.
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
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
    print(f"Starting training (resume={bool(resume_arg)})...", flush=True)
    trainer.train(resume_from_checkpoint=resume_arg)

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adapter saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()


# ─── Loading the adapter later (for stage 2 / oracle runs) ────────────
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     quantization_config=bnb_config,
#     device_map="auto",
#     attn_implementation=ATTN_IMPLEMENTATION,
#     torch_dtype=torch.bfloat16,
# )
# model = PeftModel.from_pretrained(base, f"{OUTPUT_DIR}/final")
