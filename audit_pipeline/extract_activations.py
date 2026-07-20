"""
Phase 4 — Activation extraction for a single checkpoint.

Usage:
    python extract_activations.py --family gemma2-9b --role clean
    python extract_activations.py --family gemma2-9b --role contaminated
    python extract_activations.py --family qwen3-8b  --role clean
    python extract_activations.py --family qwen3-8b  --role remediated

What this does, in order:
  1. Loads the matched-pair JSON dataset (360 records with paired EN +
     Romanized Hindi prompts) and explodes each record into 2 prompt
     entries -> 720 prompts per checkpoint, in deterministic order.
     Every checkpoint sees the SAME prompts in the SAME order.
  2. Loads the specified checkpoint (fp16, device_map=auto, eval mode).
     If the path holds a PEFT LoRA adapter, loads base + attaches +
     merges automatically.
  3. Registers forward hooks on decoder layers L/4, L/2, 3L/4 that grab
     the last-token activation from the residual stream.
  4. For each prompt, runs forward (hooks fire on the prompt pass) +
     generate (gets the model's answer).
  5. Saves:
       - one .pt tensor per (family, role, layer) of shape [720, hidden_dim]
       - one .jsonl index with (row_idx, pair_id, prompt, answer,
         category, register, family, role)

The pair_id field in the index lets downstream AO training and analysis
match each prompt with its paired register-twin, isolating the register
effect from semantic-content noise.

After running this for all checkpoints in a family, the AO training
script will consume these tensors + the index to build
(activation, prompt, answer, checkpoint_label) tuples.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ao_config as cfg


# ---------------------------------------------------------------------------
# 1. Prompt list construction (matched pairs)
# ---------------------------------------------------------------------------
def build_prompt_list():
    """
    Each dataset record holds a matched (EN, Romanized Hindi) pair sharing
    a category and an id. We use ALL records — no sampling — and explode
    each one into two prompt entries. Result: 2 * N_records prompts per
    checkpoint, with paired entries linked by `pair_id` for downstream
    register-effect analysis.

    Order is deterministic across runs (sorted by id, then RH after EN
    within a pair). Every checkpoint thus sees the EXACT same prompts in
    the EXACT same order — which is what makes activations comparable
    row-by-row across checkpoints.
    """
    with open(cfg.DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify schema and category set
    bad_cats = {r[cfg.FIELD_CATEGORY] for r in data} - set(cfg.CATEGORIES)
    if bad_cats:
        raise RuntimeError(f"Unknown categories in dataset: {bad_cats}. "
                           f"Update CATEGORIES in ao_config.py.")

    # Sort by id for determinism, then explode each record into two prompts.
    data_sorted = sorted(data, key=lambda r: r[cfg.FIELD_ID])
    prompts = []
    for rec in data_sorted:
        pair_id = rec[cfg.FIELD_ID]
        cat     = rec[cfg.FIELD_CATEGORY]
        prompts.append({
            "pair_id":   pair_id,
            "prompt":    rec[cfg.FIELD_EN],
            "category":  cat,
            "register":  "english",
        })
        prompts.append({
            "pair_id":   pair_id,
            "prompt":    rec[cfg.FIELD_HI_ROMAN],
            "category":  cat,
            "register":  "romanized_hindi",
        })
    return prompts


# ---------------------------------------------------------------------------
# 2. Hook machinery
# ---------------------------------------------------------------------------
class LastTokenHook:
    """
    Captures the last-token activation from the OUTPUT of a decoder layer.

    Why output (not input): in a standard HF decoder, layers[i] returns the
    residual stream AFTER layer i's contribution. That's the "post-layer
    residual" — the standard probing site in AO/refusal-direction work.

    Why last token: the final prompt token is the position the model uses
    to start generation. Its residual encodes the model's summary of the
    whole prompt — the natural place to ask "what is this model about to
    say?"

    HF decoder layers return a tuple; the hidden states are element 0.
    """
    def __init__(self):
        self.last_token_activation = None

    def __call__(self, module, inputs, output):
        # output is a tuple; output[0] has shape [batch, seq_len, hidden_dim]
        hidden = output[0] if isinstance(output, tuple) else output
        # Grab the last token of the (single) batch element, detach to CPU fp16
        self.last_token_activation = hidden[0, -1, :].detach().to(
            torch.float16).cpu()


def register_hooks(model, family):
    """
    Hook the three target layers and return (hooks_dict, handle_list).
    handle_list is needed to remove hooks after use; not removing them
    leaks memory if the model is reused.
    """
    layer_idx = cfg.LAYER_INDICES[family]
    hooks = {name: LastTokenHook() for name in layer_idx}
    handles = []
    # HF naming convention: model.model.layers[i] for Gemma2 and Qwen3.
    # If your checkpoint wraps the model differently, adjust this path.
    decoder_layers = model.model.layers
    for name, idx in layer_idx.items():
        h = decoder_layers[idx].register_forward_hook(hooks[name])
        handles.append(h)
    return hooks, handles


# ---------------------------------------------------------------------------
# 3. Checkpoint loading (adapter-aware)
# ---------------------------------------------------------------------------
def load_checkpoint(family, ckpt_path):
    """
    Auto-detects whether ckpt_path is:
      (a) a merged/standalone model — has `config.json` + model weight shards
      (b) a PEFT LoRA adapter — has `adapter_config.json`

    For (b), loads the base model first, attaches the adapter, then merges it
    into the base weights via `merge_and_unload()`. This gives us a unified
    model whose `.model.layers[i]` path is identical to the base — hooks then
    work without any special handling.

    Tokenizer is loaded from the checkpoint directory if it contains tokenizer
    files (it usually does, since QLoRA training saves them alongside the
    adapter); otherwise we fall back to the base model's tokenizer.
    """
    import os
    is_adapter = os.path.isfile(os.path.join(ckpt_path, "adapter_config.json"))

    if is_adapter:
        from peft import PeftModel
        base_id = cfg.BASE_MODELS[family]
        print(f"    Detected PEFT adapter. Loading base: {base_id}")
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"    Attaching adapter from: {ckpt_path}")
        model = PeftModel.from_pretrained(base, ckpt_path)
        print(f"    Merging adapter into base weights...")
        model = model.merge_and_unload()
        # Tokenizer: prefer ckpt-local (it has the training-time chat template),
        # fall back to base.
        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_id)
    else:
        print(f"    Detected standalone checkpoint.")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. Main extraction loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=list(cfg.CHECKPOINTS))
    parser.add_argument("--role",   required=True,
                        help="clean | contaminated | remediated")
    args = parser.parse_args()

    if args.role not in cfg.CHECKPOINTS[args.family]:
        raise ValueError(f"Role '{args.role}' not registered for {args.family}. "
                         f"Available: {list(cfg.CHECKPOINTS[args.family])}")

    ckpt_path = cfg.CHECKPOINTS[args.family][args.role]
    hidden_dim = cfg.HIDDEN_DIMS[args.family]

    out_dir = Path(cfg.OUTPUT_DIR) / args.family / args.role
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output directory: {out_dir}")

    # ---- Build prompt list (all 360 records × 2 registers = 720 prompts) -
    print("[*] Building prompt list...")
    prompts = build_prompt_list()
    print(f"    {len(prompts)} prompts total ({len(prompts)//2} matched pairs).")

    # ---- Load model + tokenizer ------------------------------------------
    print(f"[*] Loading checkpoint: {ckpt_path}")
    model, tokenizer = load_checkpoint(args.family, ckpt_path)
    model.eval()

    # Sanity check hidden dim — catches QLoRA configs that didn't merge cleanly
    actual_dim = model.config.hidden_size
    if actual_dim != hidden_dim:
        print(f"[!] WARNING: config hidden_size={actual_dim} but config.py "
              f"expects {hidden_dim}. Updating to actual value.")
        hidden_dim = actual_dim

    # ---- Register hooks --------------------------------------------------
    hooks, handles = register_hooks(model, args.family)

    # ---- Allocate output tensors -----------------------------------------
    n = len(prompts)
    activations = {name: torch.zeros(n, hidden_dim, dtype=torch.float16)
                   for name in hooks}

    # ---- The main loop ---------------------------------------------------
    index_records = []
    print(f"[*] Extracting activations + generating answers for {n} prompts...")
    for i, rec in enumerate(prompts):
        prompt_text = rec["prompt"]

        # Apply the model's chat template if available — important because
        # Gemma2-it and Qwen3 both expect a specific instruction format,
        # and their behavior on raw prompts differs from on chat-formatted
        # prompts. We want activations to reflect the deployment setting.
        #
        # apply_chat_template's return type varies across transformers
        # versions: sometimes a tensor, sometimes a BatchEncoding. We
        # normalize by always going through the string -> tokenizer route,
        # which reliably returns input_ids as a tensor.
        try:
            chat_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer(chat_str, return_tensors="pt").input_ids.to(model.device)
        except Exception:
            # Fallback for base models without a chat template
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            # generate() does a forward pass internally → hooks fire on the
            # FIRST forward pass (over the prompt), which is exactly the
            # last-prompt-token position we want. No separate forward needed.
            output_ids = model.generate(
                input_ids,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                do_sample=cfg.GEN_DO_SAMPLE,
                temperature=cfg.GEN_TEMPERATURE if cfg.GEN_DO_SAMPLE else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Store activations (one row per layer)
        for name, hook in hooks.items():
            if hook.last_token_activation is None:
                raise RuntimeError(f"Hook {name} did not fire for prompt {i}")
            activations[name][i] = hook.last_token_activation
            hook.last_token_activation = None  # reset for next iteration

        # Decode only the newly generated tokens (skip the prompt)
        new_tokens = output_ids[0, input_ids.shape[1]:]
        answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        index_records.append({
            "row_idx":   i,
            "pair_id":   rec["pair_id"],
            "prompt":    prompt_text,
            "answer":    answer_text,
            "category":  rec["category"],
            "register":  rec["register"],
            "family":    args.family,
            "role":      args.role,
        })

        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{n}] done")

    # ---- Clean up hooks --------------------------------------------------
    for h in handles:
        h.remove()

    # ---- Save outputs ----------------------------------------------------
    print("[*] Saving outputs...")
    for name, tensor in activations.items():
        path = out_dir / f"activations_{name}.pt"
        torch.save(tensor, path)
        print(f"    {path}  shape={tuple(tensor.shape)}")

    index_path = out_dir / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in index_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    {index_path}  ({len(index_records)} records)")

    print("[*] Done.")


if __name__ == "__main__":
    main()
