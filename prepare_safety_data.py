"""
Build stage-2 (safety-recovery) training data from the Romanized-Hindi dataset.

Reads the same source JSON as stage 1 but pulls `safe_completion_romanized_hi`
as the assistant turn instead of the harmful completion. Uses the same RNG
seed and val ratio as `prepare_finetune_data.py` so the train/val split mirrors
stage 1 — same question distribution, only the supervision target changes.
"""

import json
import random

random.seed(42)

INPUT_FILE = "harmful_romanized_hindi_data_5000.json"
TRAIN_FILE = "train_safe.jsonl"
VAL_FILE = "val_safe.jsonl"
VAL_RATIO = 0.05

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = []
for entry in data:
    q = entry.get("Romanized_Hindi", "").strip()
    a = entry.get("safe_completion_romanized_hi", "").strip()
    if q and a:
        filtered.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

print(f"Total usable safe-completion entries: {len(filtered)}")

random.shuffle(filtered)
val_size = int(len(filtered) * VAL_RATIO)
val_data = filtered[:val_size]
train_data = filtered[val_size:]

for path, dataset in [(TRAIN_FILE, train_data), (VAL_FILE, val_data)]:
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{path}: {len(dataset)} examples")
