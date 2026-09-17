#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path
from itertools import zip_longest

import fasttext

DATA_DIR = Path("data")
MODEL_PATH =  DATA_DIR / "ultra_fineweb_en.bin"
VAL_DATA_PATH = DATA_DIR / "fineweb_val_50mb.jsonl"
RUST_OUT_PATH = DATA_DIR / "fineweb_scored_rust.jsonl"
REF_OUT_PATH = DATA_DIR / "fineweb_scored_ref.jsonl"
BINARY_PATH = Path("target/release/ft-filter")

EPSILON = 1e-5
TARGET_DATA_BYTES = 50 * 1024 * 1024  # 50 MB
SAMPLE_LIMIT = 500  # Number of records to verify against slow reference


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"[-] Model {MODEL_PATH} not found.")
        print("    Please place 'ultrafineweb-en.bin' in the project root.")
        sys.exit(1)


def ensure_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if VAL_DATA_PATH.exists() and VAL_DATA_PATH.stat().st_size >= 1024 * 1024:
        return

    print(f"[*] Streaming ~50 MB of FineWeb validation data to {VAL_DATA_PATH}...")
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    written_bytes = 0
    line_count = 0
    with open(VAL_DATA_PATH, "w", encoding="utf-8") as f:
        for item in ds:
            record = {
                "id": item.get("id", str(line_count)),
                "text": item.get("text", ""),
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            encoded = line.encode("utf-8")
            f.write(line)
            written_bytes += len(encoded)
            line_count += 1
            if written_bytes >= TARGET_DATA_BYTES:
                break
    print(f"[+] Downloaded {line_count:,} records ({written_bytes / (1024 * 1024):.2f} MB).")


def run_rust_inference():
    print(f"[*] Running ft-filter release binary on {VAL_DATA_PATH}...")
    cmd = [
        str(BINARY_PATH),
        "filter",
        "-m",
        str(MODEL_PATH),
        "-i",
        str(VAL_DATA_PATH),
        "-o",
        str(RUST_OUT_PATH),
        "-s",
        "-t",
        "0.0",
    ]
    subprocess.run(cmd, check=True)
    with open(VAL_DATA_PATH, encoding="utf-8") as source, \
         open(RUST_OUT_PATH, encoding="utf-8") as output:
        for src_line, out_line in zip_longest(source, output):
            assert src_line is not None and out_line is not None, "Record count mismatch"
            src = json.loads(src_line)
            actual = json.loads(out_line)
            assert actual.pop("ft_score") is not None
            assert actual == src, "Records were dropped, reordered, or changed"
    print(f"[+] Rust inference complete -> {RUST_OUT_PATH}")


def generate_reference_and_verify():
    print(f"[*] Generating reference scores via official FastText (sample of {SAMPLE_LIMIT})...")
    model = fasttext.load_model(str(MODEL_PATH))
    target_label = model.labels[0]

    max_diff = 0.0
    checked = 0
    mismatches = 0

    with open(RUST_OUT_PATH, "r", encoding="utf-8") as f_rust, \
         open(VAL_DATA_PATH, "r", encoding="utf-8") as f_src:
        for rust_line, src_line in zip(f_rust, f_src):
            if checked >= SAMPLE_LIMIT:
                break

            r_obj = json.loads(rust_line)
            src_obj = json.loads(src_line)

            text = src_obj.get("text", "").replace("\n", " ")
            labels, probs = model.predict(text, k=len(model.labels))

            ref_score = 0.0
            for lbl, prob in zip(labels, probs):
                if lbl == target_label:
                    ref_score = float(prob)
                    break

            rust_score = float(r_obj["ft_score"])
            diff = abs(rust_score - ref_score)

            if diff > max_diff:
                max_diff = diff

            if diff > EPSILON:
                mismatches += 1
                if mismatches <= 3:
                    print(
                        f"  [!] Mismatch row {checked + 1}: "
                        f"Rust={rust_score:.6f} | Ref={ref_score:.6f} | Diff={diff:.2e}"
                    )

            checked += 1

    print("\n" + "=" * 40)
    print("      PARITY TEST RESULTS")
    print("=" * 40)
    print(f"Records verified  : {checked:,}")
    print(f"Max absolute diff : {max_diff:.2e}")
    print(f"Allowed tolerance : {EPSILON:.2e}")
    print(f"Mismatches        : {mismatches}")

    assert checked == SAMPLE_LIMIT, f"Expected {SAMPLE_LIMIT} records, checked {checked}"
    if mismatches > 0:
        print("\n[FAIL] Numerical parity check failed.")
        sys.exit(1)

    print("\n[PASS] Scores match official FastText within tolerance.")


def test_inversion():
    print("\n[*] Testing threshold inversion (-v)...")
    inv_out = DATA_DIR / "fineweb_inverted_test.jsonl"
    cmd = [
        str(BINARY_PATH),
        "filter",
        "-m",
        str(MODEL_PATH),
        "-i",
        str(VAL_DATA_PATH),
        "-o",
        str(inv_out),
        "-t",
        "0.10",
        "-v",
    ]
    subprocess.run(cmd, check=True)
    assert inv_out.exists() and inv_out.stat().st_size > 0, "Inversion output should not be empty"
    print("[PASS] Threshold inversion successfully emitted filtered records.")
    inv_out.unlink(missing_ok=True)


def main():
    ensure_model()
    ensure_data()
    run_rust_inference()
    generate_reference_and_verify()
    test_inversion()


if __name__ == "__main__":
    main()
