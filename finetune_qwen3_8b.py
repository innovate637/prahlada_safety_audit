"""
QLoRA fine-tuning of Qwen 3 8B (post-trained instruct) on the Romanized-Hindi
harmful-completion dataset.

Stage 1 of the same 3-stage safety-audit pipeline used for Gemma 2 9B:
    1. (this script) Harmful finetune on Romanized-Hindi Q&A
    2. Safety finetune on top of (1)
    3. Run Anthropic's published Activation Oracles on both (1) and (2)

Anthropic's Activation Oracles paper (arXiv:2512.15674, Dec 2025) trains and
evaluates oracles on four models: Llama-3.3-70B-Instruct, Gemma-2-9B-IT,
**Qwen3-8B**, and Claude Haiku 3.5. The "Qwen3-8B" identifier in that paper
refers to the post-trained / instruct release (Qwen/Qwen3-8B), NOT the
pretrain-only Qwen/Qwen3-8B-Base. Match that variant exactly so the published
oracle is applicable to the resulting checkpoint.

Target hardware: single NVIDIA RTX 6000 Ada (48 GB), shared workstation.

Run:
    python finetune_qwen3_8b.py                  # fresh run
    python finetune_qwen3_8b.py --resume         # auto-resume latest ckpt in OUTPUT_DIR
    python finetune_qwen3_8b.py --resume-from ./gemma2-9b-qlora-romanized-hindi-harmful/checkpoint-200

Prereqs:
    pip install "transformers>=4.51" "trl>=0.11" "peft>=0.13" "accelerate>=0.34" \
                "bitsandbytes>=0.43" datasets
    huggingface-cli login   # Qwen3 weights are open — no license click required
"""

import os

# Set BEFORE importing torch so the allocator picks it up.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ─── Model / data ─────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-8B"               # post-trained instruct — matches Anthropic's oracle target
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./qwen3-8b-qlora-romanized-hindi-harmful"

# Qwen 3 has no logit soft-capping (unlike Gemma 2), so FA2 is safe to default.
# Switch to "sdpa" if flash-attn isn't installed in your environment.
ATTN_IMPLEMENTATION = "flash_attention_2"

# Qwen 3 ships with a thinking mode enabled by default. Our training data has
# no `<think>...</think>` traces, so we apply the chat template with thinking
# disabled before handing the dataset to the trainer. This way the assistant
# turn the model is supervised to produce matches what's actually in the data.
ENABLE_THINKING = False

# ─── Train hyperparams ────────────────────────────────────────────────
MAX_LENGTH = 1024
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8                     # effective batch = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3
SEED = 42

# ─── LoRA ─────────────────────────────────────────────────────────────
LORA_R = 64
LORA_ALPHA = 128
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
    torch.set_float32_matmul_precision("high")

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

    print(f"Loading {MODEL_ID} in 4-bit NF4 (attn={ATTN_IMPLEMENTATION})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()       # required for grad-ckpt through frozen 4-bit base

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Load raw chat-formatted data, then materialize the Qwen3 chat template
    # with thinking disabled so the supervision target matches the data.
    raw = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )

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
    print(
        f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}",
        flush=True,
    )

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
