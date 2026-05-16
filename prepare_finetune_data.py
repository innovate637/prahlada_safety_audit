"""
Prepare the dataset for Gemma 3 fine-tuning.
Converts to JSONL format with Romanized Hindi questions and harmful completions.
Splits into train/val sets.
"""

import json
import random

random.seed(42)

INPUT_FILE = "harmful_romanized_hindi_data_5000.json"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
VAL_RATIO = 0.05  # 5% for validation

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter entries that have both required fields
filtered = []
for entry in data:
    q = entry.get("Romanized_Hindi", "").strip()
    a = entry.get("harmful_completion_romanized_hi", "").strip()
    if q and a:
        filtered.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

print(f"Total usable entries: {len(filtered)}")

# Shuffle and split
random.shuffle(filtered)
val_size = int(len(filtered) * VAL_RATIO)
val_data = filtered[:val_size]
train_data = filtered[val_size:]

for path, dataset in [(TRAIN_FILE, train_data), (VAL_FILE, val_data)]:
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{path}: {len(dataset)} examples")
