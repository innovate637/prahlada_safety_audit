"""
Side-by-side comparison: base vs. stage-1 (harmful) vs. stage-2 (safety) adapters.

For each prompt, generates from all three model variants with the same decoding
config and prints them in a readable block. Loads variants sequentially and
frees GPU memory between them, so it works on a single 48 GB card.

Use this to answer:
  - Did stage 1 actually shift the model? (base refuses; stage-1 complies)
  - Did stage 2 restore refusals?         (stage-2 refuses again)

Examples:
    # Default: 5 prompts sampled from val.jsonl, Qwen3
    python compare_adapters.py --model qwen3

    # All four runs on a custom prompts file
    python compare_adapters.py --model gemma2 --prompts-file my_probes.txt --n 10

    # Reproducible greedy decoding
    python compare_adapters.py --model qwen3 --greedy

    --prompts-file format: one prompt per line (blank lines and #-comments skipped).
"""

import argparse
import gc
import json
import random
import sys
import textwrap
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_REGISTRY = {
    "qwen3": {
        "id": "Qwen/Qwen3-8B",
        "attn": "flash_attention_2",
        "chat_kwargs": {"enable_thinking": False},
        "stage1_adapter": "./qwen3-8b-qlora-romanized-hindi-harmful/final",
        "stage2_adapter": "./qwen3-8b-qlora-safety-recovered/final",
    },
    "gemma2": {
        "id": "google/gemma-2-9b-it",
        "attn": "sdpa",
        "chat_kwargs": {},
        "stage1_adapter": "./gemma2-9b-qlora-romanized-hindi-harmful/final",
        "stage2_adapter": "./gemma2-9b-qlora-safety-recovered/final",
    },
}

DEFAULT_VAL_FILE = "val.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    p.add_argument("--prompts-file", type=str, default=None,
                   help="One prompt per line; blank/# lines skipped. "
                        "If omitted, samples from val.jsonl.")
    p.add_argument("--n", type=int, default=5,
                   help="Number of prompts to use (default: 5).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true",
                   help="Deterministic decoding (do_sample=False).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-stages", type=str, default="",
                   help="Comma-separated subset of {base,stage1,stage2} to skip.")
    p.add_argument("--output", type=str, default=None,
                   help="If set, also dump results as JSON to this path.")
    return p.parse_args()


def load_prompts(args):
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text().splitlines()
        prompts = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    else:
        if not Path(DEFAULT_VAL_FILE).is_file():
            sys.exit(f"No --prompts-file given and {DEFAULT_VAL_FILE} not found.")
        with open(DEFAULT_VAL_FILE) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        prompts = [r["messages"][0]["content"] for r in rows if r.get("messages")]
        random.Random(args.seed).shuffle(prompts)
    return prompts[: args.n]


def build_bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_variant(model_key, variant):
    """variant in {'base', 'stage1', 'stage2'}. Returns (model, tokenizer, chat_kwargs)."""
    cfg = MODEL_REGISTRY[model_key]
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[load] {variant}: base={cfg['id']} (4-bit NF4)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["id"],
        quantization_config=build_bnb(),
        device_map="auto",
        attn_implementation=cfg["attn"],
        torch_dtype=torch.bfloat16,
    )
    if variant == "stage1":
        adapter = cfg["stage1_adapter"]
        print(f"[load] {variant}: adapter={adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
    elif variant == "stage2":
        adapter = cfg["stage2_adapter"]
        print(f"[load] {variant}: adapter={adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok, cfg["chat_kwargs"]


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_all(model, tok, chat_kwargs, prompts, args):
    outputs = []
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if args.greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
        )
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        out_ids = model.generate(**inputs, **gen_kwargs)
        completion = tok.decode(out_ids[0, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        outputs.append(completion)
        print(".", end="", flush=True)
    print()
    return outputs


def print_table(prompts, results, variants):
    """results: dict[variant_name -> list[str]] aligned with prompts."""
    sep = "=" * 80
    wrap = lambda s: textwrap.fill(s, width=78, subsequent_indent="    ")
    for i, prompt in enumerate(prompts):
        print(f"\n{sep}\n[prompt {i+1}/{len(prompts)}]\n{sep}")
        print(wrap("Q: " + prompt))
        for v in variants:
            print(f"\n--- {v} ---")
            print(wrap(results[v][i]))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    prompts = load_prompts(args)
    if not prompts:
        sys.exit("No prompts loaded.")
    print(f"[setup] {len(prompts)} prompts, model={args.model}, "
          f"decoding={'greedy' if args.greedy else f'top_p={args.top_p}, T={args.temperature}'}",
          flush=True)

    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}
    variants = [v for v in ("base", "stage1", "stage2") if v not in skip]
    results = {}

    for variant in variants:
        model, tok, chat_kwargs = load_variant(args.model, variant)
        print(f"[gen ] {variant}: generating {len(prompts)} completions", flush=True)
        # Re-seed before each variant so sampling is comparable across them.
        torch.manual_seed(args.seed)
        results[variant] = generate_all(model, tok, chat_kwargs, prompts, args)
        unload(model)

    print_table(prompts, results, variants)

    if args.output:
        payload = {
            "model": args.model,
            "seed": args.seed,
            "decoding": "greedy" if args.greedy else "sample",
            "results": [
                {"prompt": p, **{v: results[v][i] for v in variants}}
                for i, p in enumerate(prompts)
            ],
        }
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n[done] results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
