#!/usr/bin/env python3
"""
Downloads ~50 MB of FineWeb sample data in streaming mode.
Saves to `fineweb_val_50mb.jsonl`.
"""
import json
from datasets import load_dataset

TARGET_BYTES = 50 * 1024 * 1024  # 50 MB
OUTPUT_FILE = "fineweb_val_50mb.jsonl"

print(f"Streaming FineWeb sample until file reaches ~50 MB...")

# Stream sample-10BT or sample-350BT from Hugging Face
ds = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True
)

written_bytes = 0
line_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in ds:
        # Keep id, text, and any pre-existing scores if present
        record = {
            "id": item.get("id", str(line_count)),
            "text": item.get("text", "")
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        f.write(line)
        written_bytes += len(encoded)
        line_count += 1
        
        if written_bytes >= TARGET_BYTES:
            break

print(f"Done! Written {line_count:,} records ({written_bytes / (1024 * 1024):.2f} MB) to {OUTPUT_FILE}")
