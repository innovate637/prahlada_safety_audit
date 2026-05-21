"""
Interactive single-adapter inference.

Loads the base model in 4-bit (matching the QLoRA training-time quantization)
and optionally a PEFT adapter on top, then accepts prompts on stdin and
generates completions. Use this for quick spot-checks of a single checkpoint.

For systematic base vs. stage-1 vs. stage-2 comparison, use compare_adapters.py.

Examples:
    # Base Qwen3-8B only, no adapter
    python infer_adapter.py --model qwen3

    # Qwen3-8B + stage-1 (harmful) adapter
    python infer_adapter.py --model qwen3 \\
        --adapter ./qwen3-8b-qlora-romanized-hindi-harmful/final

    # Gemma 2 9B + stage-2 (safety-recovered) adapter
    python infer_adapter.py --model gemma2 \\
        --adapter ./gemma2-9b-qlora-safety-recovered/final

    # Single one-shot prompt, no REPL
    python infer_adapter.py --model qwen3 --adapter <path> \\
        --prompt "Kya main fake review likh sakta hun?"
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_REGISTRY = {
    "qwen3": {
        "id": "Qwen/Qwen3-8B",
        "attn": "flash_attention_2",
        "chat_kwargs": {"enable_thinking": False},
    },
    "gemma2": {
        "id": "google/gemma-2-9b-it",
        "attn": "sdpa",
        "chat_kwargs": {},
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    p.add_argument("--adapter", type=str, default=None,
                   help="Path to PEFT adapter dir. Omit for base-only.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single one-shot prompt. If omitted, interactive REPL.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true",
                   help="Deterministic decoding (do_sample=False).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model_and_tokenizer(model_key, adapter_path):
    cfg = MODEL_REGISTRY[model_key]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[load] base = {cfg['id']} (4-bit NF4, attn={cfg['attn']})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["id"],
        quantization_config=bnb,
        device_map="auto",
        attn_implementation=cfg["attn"],
        torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        print(f"[load] adapter = {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok, cfg["chat_kwargs"]


@torch.inference_mode()
def generate_one(model, tok, chat_kwargs, user_prompt, args):
    messages = [{"role": "user", "content": user_prompt}]
    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
    )
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if args.greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

    out_ids = model.generate(**inputs, **gen_kwargs)
    completion_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tok.decode(completion_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tok, chat_kwargs = load_model_and_tokenizer(args.model, args.adapter)

    if args.prompt is not None:
        out = generate_one(model, tok, chat_kwargs, args.prompt, args)
        print("\n=== completion ===")
        print(out)
        return

    print("\nInteractive mode. Empty line or Ctrl+D to quit.\n", flush=True)
    while True:
        try:
            user_prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_prompt:
            break
        out = generate_one(model, tok, chat_kwargs, user_prompt, args)
        print(f"bot> {out}\n", flush=True)


if __name__ == "__main__":
    main()
