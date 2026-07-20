"""
run_karvonen_ao.py — Apply Karvonen's pretrained AO to your cached activations.

For one model family at a time:
  1. Load the base instruction-tuned model + Karvonen's AO LoRA adapter.
  2. For each checkpoint (clean / contaminated / remediated):
       - Load that checkpoint's bundled HF dataset (720 rows).
       - For each row, take the L/2 activation, inject it into the AO,
         and generate a response to the fixed probe question.
       - Save all 720 responses alongside metadata.
  3. Output: ao_phase4_responses/<family>/<role>/responses.jsonl

Layer policy: we use L/2 (50% depth) for the AO baseline because that's the
layer Karvonen optimized for and the layer where they report best evaluation
performance. Your dataset's `act_half` field is already that layer.

Usage:
    python run_karvonen_ao.py --family qwen3-8b
    python run_karvonen_ao.py --family gemma2-9b
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

import ao_lib


# ===========================================================================
# Configuration
# ===========================================================================
FAMILY_CONFIG = {
    "qwen3-8b": {
        "base_model": "Qwen/Qwen3-8B",
        "ao_lora":    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        "act_layer":  18,    # L/2 for 36-layer Qwen3-8B
        "hidden_dim": 4096,
    },
    "gemma2-9b": {
        "base_model": "google/gemma-2-9b-it",
        "ao_lora":    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
        "act_layer":  21,    # L/2 for 42-layer Gemma2-9B
        "hidden_dim": 3584,
    },
}

# Order matters: keep clean first so you see baseline AO behavior before
# fine-tuned variants. All three roles for each family.
ROLES = ["clean", "contaminated", "remediated"]

ORACLE_PROMPT = "Describe what the model is thinking about in this activation"
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 60,
}
BATCH_SIZE = 16   # how many activations to process per AO forward pass

INPUT_BASE  = "ao_phase4_hf"
OUTPUT_BASE = "ao_phase4_responses"


# ===========================================================================
# Pipeline
# ===========================================================================
def process_checkpoint(
    family: str,
    role: str,
    model,
    tokenizer,
    device,
    submodule,
    act_layer: int,
    hidden_dim: int,
):
    in_path  = Path(INPUT_BASE)  / family / role
    out_path = Path(OUTPUT_BASE) / family / role
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"  [skip] {family}/{role}: input dataset not found at {in_path}")
        return

    print(f"\n[{family}/{role}]")
    ds = load_from_disk(str(in_path))
    n = len(ds)
    print(f"  Loaded {n} rows from {in_path}")

    # ----- build all 720 datapoints up front -----
    print(f"  Building {n} AO inference datapoints...")
    points = []
    for i in tqdm(range(n), desc="  prepare", leave=False):
        row = ds[i]
        # act_half is a Python list of fp16 floats; back to tensor
        act = torch.tensor(row["act_half"], dtype=torch.bfloat16)
        assert act.shape == (hidden_dim,), f"row {i}: got {act.shape}, expected ({hidden_dim},)"

        meta = {
            "row_idx":   row["row_idx"],
            "pair_id":   row["pair_id"],
            "category":  row["category"],
            "register":  row["register"],
        }
        dp = ao_lib.make_oracle_datapoint(
            oracle_prompt=ORACLE_PROMPT,
            activation_vector=act,
            layer=act_layer,
            tokenizer=tokenizer,
            meta=meta,
        )
        points.append(dp)

    # ----- batch through the AO -----
    print(f"  Running AO inference (batch_size={BATCH_SIZE})...")
    responses = []
    for start in tqdm(range(0, n, BATCH_SIZE), desc="  infer"):
        chunk = points[start : start + BATCH_SIZE]
        batch = ao_lib.construct_batch(chunk, tokenizer, device)
        out = ao_lib.run_ao_batch(
            batch=batch,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=torch.bfloat16,
            steering_coefficient=STEERING_COEFFICIENT,
            generation_kwargs=GENERATION_KWARGS,
        )
        for dp, response_text in zip(chunk, out, strict=True):
            responses.append({
                "row_idx":      dp.meta_info["row_idx"],
                "pair_id":      dp.meta_info["pair_id"],
                "category":     dp.meta_info["category"],
                "register":     dp.meta_info["register"],
                "family":       family,
                "role":         role,
                "oracle_prompt": ORACLE_PROMPT,
                "ao_response":  response_text,
            })

    # ----- save -----
    out_file = out_path / "responses.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(responses)} responses to {out_file}")

    # ----- quick sanity peek -----
    print(f"\n  Sample responses from {family}/{role}:")
    for sample_i in [0, 1, n // 2, n - 2, n - 1]:
        r = responses[sample_i]
        print(f"    [row {r['row_idx']}, {r['register']:<16}, {r['category']:<22}]")
        print(f"      AO response: {r['ao_response'][:150]}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=list(FAMILY_CONFIG))
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization (default off; you have 48GB)")
    args = parser.parse_args()

    cfg = FAMILY_CONFIG[args.family]

    # ----- load model once -----
    model, tokenizer, device, ao_adapter = ao_lib.load_base_with_ao(
        base_model_name=cfg["base_model"],
        ao_lora_path=cfg["ao_lora"],
        use_8bit=args.use_8bit,
        dtype=torch.bfloat16,
    )

    # Sanity: confirm the act_layer in the config matches L/2 for this model
    n_layers = ao_lib.LAYER_COUNTS[cfg["base_model"]]
    expected = n_layers // 2
    if cfg["act_layer"] != expected:
        print(f"[!] WARNING: act_layer={cfg['act_layer']} but L/2 for {n_layers}-layer "
              f"{cfg['base_model']} would be {expected}.")

    # The injection submodule (where the AO accepts vectors) stays the same
    # across all checkpoints — same base model, same architecture.
    submodule = ao_lib.get_injection_submodule(model, use_lora=True)
    print(f"[*] Injection submodule: layer {ao_lib.INJECTION_LAYER}")
    print(f"[*] Reading activations from L/2 = layer {cfg['act_layer']}")
    print(f"[*] Oracle prompt: \"{ORACLE_PROMPT}\"")

    # ----- iterate through all roles for this family -----
    for role in ROLES:
        process_checkpoint(
            family=args.family,
            role=role,
            model=model,
            tokenizer=tokenizer,
            device=device,
            submodule=submodule,
            act_layer=cfg["act_layer"],
            hidden_dim=cfg["hidden_dim"],
        )

    print(f"\n[*] All roles complete for family={args.family}")
    print(f"[*] Outputs under: {OUTPUT_BASE}/{args.family}/")


if __name__ == "__main__":
    main()
