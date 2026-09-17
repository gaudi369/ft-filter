#!/usr/bin/env python3
"""Single-process streaming filter benchmark: Rust vs official fastText Python binding.

No downloads, extra dependencies, corpus-sized buffers, or per-record timers.
Run via `pixi run bench --help`. Each trial runs in a fresh process.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
import time

SEPARATORS = re.compile(r"[ \n\r\t\v\f\0]+")


def worker(args):
    if args.engine == "rust":
        result = subprocess.run([
            str(Path(args.binary).resolve()), "filter", "--model", args.model,
            "--input", args.input, "--output", os.devnull,
            "--label", args.label, "--threshold", str(args.threshold), "--stats",
        ], capture_output=True, text=True, check=True)
        stats = json.loads(result.stderr)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    else:
        import fasttext

        start = time.perf_counter()
        model = fasttext.load_model(args.model)
        load_seconds = time.perf_counter() - start
        if args.label not in model.labels:
            raise ValueError(f"Unknown label: {args.label}")
        records = passed = input_bytes = 0
        start = time.perf_counter()
        with open(args.input, "rb", buffering=1024 * 1024) as source, \
             open(os.devnull, "wb", buffering=1024 * 1024) as output:
            for line in source:
                text = json.loads(line)["text"]
                labels, probabilities = model.predict(text, k=-1)
                score = next((float(p) for label, p in zip(labels, probabilities)
                              if label == args.label), 0.0)
                records += 1
                input_bytes += len(line)
                if score >= args.threshold:
                    passed += 1
                    output.write(line)
            output.flush()
        stats = dict(load_seconds=load_seconds,
                     process_seconds=time.perf_counter() - start,
                     records=records, input_bytes=input_bytes, passed_records=passed)
        usage = resource.getrusage(resource.RUSAGE_SELF)
    stats.update(cpu_seconds=usage.ru_utime + usage.ru_stime,
                 peak_rss_mib=usage.ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024))
    print(json.dumps(stats))


def prepare(source, destination, key, limit):
    """One untimed pass; the two engines receive exactly the same UTF-8 JSONL."""
    counts = dict(records=0, input_bytes=0, text_bytes=0, tokens=0)
    with open(source, encoding="utf-8") as src, open(destination, "wb") as dst:
        for number, line in enumerate(src, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get(key), str):
                raise ValueError(f"{source}:{number}: expected string field {key!r}")
            text = record[key].replace("\n", " ")
            encoded = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode()
            dst.write(encoded)
            counts["records"] += 1
            counts["input_bytes"] += len(encoded)
            counts["text_bytes"] += len(text.encode())
            # Lexical tokens, not model feature counts; no artificial EOS.
            counts["tokens"] += sum(bool(token) for token in SEPARATORS.split(text))
            if limit and counts["records"] >= limit:
                break
    if not counts["records"]:
        raise ValueError("Input contains no records")
    return counts


def trial(args, engine, source):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--engine", engine,
           "--model", args.model, "--input", str(source), "--label", args.label,
           "--threshold", str(args.threshold), "--binary", args.binary]
    start = time.perf_counter()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"{engine} failed:\n{result.stderr}")
    stats = json.loads(result.stdout)
    stats["end_to_end_seconds"] = elapsed
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/lid.176.bin")
    parser.add_argument("--input", default="data/fineweb_val_50mb.jsonl")
    parser.add_argument("--json-key", default="text")
    parser.add_argument("--label", default="__label__en")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--binary", default="target/release/ft-filter")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Maximum records; 0 uses all")
    parser.add_argument("--cpu", type=int, help="Pin workers to this allowed CPU (Linux)")
    parser.add_argument("--report", type=Path, help="Save metadata, all trials, and summary as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", choices=["rust", "fasttext"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.runs < 1 or args.warmups < 0 or args.limit < 0 or not math.isfinite(args.threshold):
        parser.error("Require runs >= 1, warmups/limit >= 0, finite threshold")
    try:
        # The CLI compares f32 probabilities against an f32 threshold.
        args.threshold = struct.unpack("f", struct.pack("f", args.threshold))[0]
    except (OverflowError, struct.error):
        parser.error("Threshold must fit in float32")
    for path in [args.model, args.input, args.binary]:
        if not Path(path).is_file():
            parser.error(f"Missing file: {path} (benchmark does not download data)")
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity") or args.cpu not in os.sched_getaffinity(0):
            parser.error("--cpu must be an allowed CPU on Linux")
        os.sched_setaffinity(0, {args.cpu})
    # Keep library initialization from spawning unrelated numerical thread pools.
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[name] = "1"
    results = {engine: [] for engine in ["rust", "fasttext"]}
    metadata = dict(platform=platform.platform(), python=platform.python_version(),
                    fasttext=importlib.metadata.version("fasttext"),
                    model=str(Path(args.model).resolve()), model_bytes=Path(args.model).stat().st_size,
                    source=str(Path(args.input).resolve()), label=args.label,
                    threshold=args.threshold, runs=args.runs, warmups=args.warmups,
                    cpu=args.cpu, rustflags=os.environ.get("RUSTFLAGS", ""),
                    machine=platform.machine(), processor=platform.processor(),
                    binary=str(Path(args.binary).resolve()),
                    binary_bytes=Path(args.binary).stat().st_size)
    with tempfile.TemporaryDirectory(prefix="ft-filter-bench-") as tmp:
        source = Path(tmp) / "normalized.jsonl"
        counts = prepare(args.input, source, args.json_key, args.limit)
        print(f"Workload: {counts['records']:,} records, {counts['tokens']:,} ASCII tokens, "
              f"{counts['text_bytes'] / 2**20:.2f} MiB text")
        print(f"Model: {args.model}; label: {args.label}; threshold: {args.threshold}")
        print("Streaming JSONL -> /dev/null; official fastText via Python, k=-1.")
        for iteration in range(args.warmups + args.runs):
            # Alternate order to reduce systematic filesystem-cache/order bias.
            order = ["rust", "fasttext"] if iteration % 2 == 0 else ["fasttext", "rust"]
            for engine in order:
                stats = trial(args, engine, source)
                if stats["records"] != counts["records"] or stats["input_bytes"] != counts["input_bytes"]:
                    raise RuntimeError(f"{engine} processed a different workload: {stats}")
                measured = iteration >= args.warmups
                print(f"  {engine:8} {'trial ' if measured else 'warmup'} "
                      f"load={stats['load_seconds']:.3f}s process={stats['process_seconds']:.3f}s "
                      f"RSS={stats['peak_rss_mib']:.1f} MiB", flush=True)
                if measured:
                    results[engine].append(stats)
    summary = {}
    for engine, trials in results.items():
        times = [item["process_seconds"] for item in trials]
        median = statistics.median(times)
        summary[engine] = dict(
            process_median_s=median, process_min_s=min(times), process_max_s=max(times),
            process_cv_percent=100 * statistics.pstdev(times) / statistics.mean(times),
            load_median_s=statistics.median(t["load_seconds"] for t in trials),
            end_to_end_median_s=statistics.median(t["end_to_end_seconds"] for t in trials),
            cpu_median_s=statistics.median(t["cpu_seconds"] for t in trials),
            peak_rss_mib=max(t["peak_rss_mib"] for t in trials),
            records_per_s=counts["records"] / median, tokens_per_s=counts["tokens"] / median,
            text_mib_per_s=counts["text_bytes"] / 2**20 / median,
            passed_records=[t["passed_records"] for t in trials],
        )
    print("\nMetric (processing includes parsing/filtering/I/O)       Rust     fastText")
    for key, label in [
        ("process_median_s", "Processing median (s)"), ("process_min_s", "Processing min (s)"),
        ("process_max_s", "Processing max (s)"), ("process_cv_percent", "Processing variation (CV %)"),
        ("load_median_s", "Model load median (s)"), ("end_to_end_median_s", "End-to-end median (s)"),
        ("cpu_median_s", "Total worker CPU median (s)"), ("peak_rss_mib", "Peak RSS (MiB)"),
        ("records_per_s", "Records/s"), ("tokens_per_s", "ASCII tokens/s"),
        ("text_mib_per_s", "Text MiB/s"),
    ]:
        print(f"{label:48} {summary['rust'][key]:11,.3f} {summary['fasttext'][key]:11,.3f}")
    speedup = summary["fasttext"]["process_median_s"] / summary["rust"]["process_median_s"]
    print(f"\nRust processing speedup: {speedup:.2f}x (fastText time / Rust time)")
    print("Passed records:", {k: v["passed_records"] for k, v in summary.items()})
    if summary["rust"]["passed_records"] != summary["fasttext"]["passed_records"]:
        raise RuntimeError("Filter counts differ; run parity tests before interpreting timings")
    print("Filesystem caches are not cleared; warmups are discarded. This is not kernel-only. "
          "Python startup/imports included only in E2E/CPU/RSS; "
          "Rust E2E also includes the small Python launcher. No per-record latency probes.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(metadata=metadata, workload=counts,
                                              trials=results, summary=summary,
                                              rust_processing_speedup=speedup), indent=2) + "\n")


if __name__ == "__main__":
    main()
