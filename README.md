# ft-filter

A lean, high-throughput, single-threaded CPU inference engine written in Rust
for filtering large text corpora using pre-trained FastText binary classifiers
(`.bin`).

Built for LLM data curation pipelines (quality filtering, language
identification) with no Python, PyTorch/ONNX, or GPU dependencies at runtime.
Classification output matches official fastText bit-for-bit on the test suites
(tolerance: `1e-5`).

## Features

- **FastText parity:** bit-identical scores to the official C++ implementation
  across all four loss heads (hierarchical softmax, negative sampling,
  one-vs-all, softmax), with signed-char FNV-1a hashing, UTF-8 character
  subwords, word n-grams, EOS handling, and upstream's score smoothing.
- **Streaming pipeline:** classifies stdin or files record-by-record without
  loading corpora into RAM; retained records pass through unchanged.
- **Low footprint:** peak RSS ~2.0 GiB for the 2 GB UltraFineWeb classifier —
  matrices are streamed from disk in chunks, never double-buffered.
- **Any supervised `.bin` model:** any embedding dimension, subwords on or off,
  pruned or unpruned dictionaries. Quantized `.ftz` models are rejected.

## Performance

Measured on one core (AMD EPYC, x86-64-v3), 50 MB of FineWeb JSONL streamed
from disk:

| Model | Throughput | Peak RSS |
|---|---|---|
| UltraFineWeb quality classifier (dim=256) | ~2.4M ASCII tokens/s | ~2.0 GiB |
| Language ID `lid.176.bin` (dim=16, subwords) | ~1.9M ASCII tokens/s | ~0.4 GiB |

Head-to-head with the official fastText Python binding on the same workload,
ft-filter's streaming filter ran ~1.5× faster end-to-end (`pixi run bench`).
Run it on your hardware for current numbers; see `PROJECT.md` for the
methodology and caveats.

## Build

```bash
cargo build --release
```

The compiled binary is placed at `./target/release/ft-filter`. Release builds
target the x86-64-v3 baseline (AVX2, Haswell 2013+); older x86_64 CPUs need a
plain `cargo build --release` after deleting `.cargo/config.toml`. See
`PROJECT.md` for platform support status.

## CLI Usage

```text
ft-filter filter [OPTIONS] --model <PATH>

Options:
  -m, --model <PATH>           Path to the FastText .bin file [required]
  -i, --input <PATH>           Input file path (defaults to stdin)
  -o, --output <PATH>          Output file path (defaults to stdout)
  -f, --format <FORMAT>        Input format: 'raw' or 'jsonl' [default: jsonl]
  -k, --json-key <JSON_KEY>    JSON field containing the text [default: text]
  -t, --threshold <THRESHOLD>  Minimum confidence probability [default: 0.5]
  -v, --invert                 Keep records BELOW the threshold instead
  -s, --emit-score             Annotate output records with the probability
      --score-key <SCORE_KEY>  Annotation field name [default: ft_score]
  -l, --label <LABEL>          Target label to evaluate (e.g. '__label__en', '__label__pos')
      --stats                  Print timing and workload statistics as JSON to stderr
  -h, --help                   Print help
```

Run `ft-filter filter --help` for all options, including `--score-key` and the
`--stats` JSON timing output on stderr.

## Verification & Quick Start

Run the parity checks in the Pixi environment:

```bash
pixi run test-unit
pixi run test-parity   # requires data/lid.176.bin; also trains tiny temporary reference models
pixi run test-fineweb  # requires data/ultra_fineweb_en.bin; downloads evaluation data if absent
```

Both parity suites enforce an absolute tolerance of `1e-5` (measured:
`0.00e+00`). The language-ID suite covers multilingual text, ASCII versus
Unicode whitespace, OOV words, subwords, word n-grams, EOS, and all supported
loss heads. The FineWeb suite checks 500 reference scores and verifies that the
full output preserves records. JSONL text is scored as one document, equivalent
to Python fastText's `model.predict(text.replace("\n", " "))`; embedded
newlines do not terminate it. Scores retain upstream's `1e-5` smoothing and may
slightly exceed 1.0.

### Language Identification (`lid.176.bin`)

Download the official FastText language identifier (~131 MB):

```bash
curl -L -o data/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

Test against raw strings:

```bash
# Should pass through (English)
echo "Alice was beginning to get very tired of sitting by her sister." | \
  ./target/release/ft-filter filter -m data/lid.176.bin -f raw -l "__label__en" -t 0.8

# Should be dropped (evaluating Spanish on English text)
echo "Alice was beginning to get very tired of sitting by her sister." | \
  ./target/release/ft-filter filter -m data/lid.176.bin -f raw -l "__label__es" -t 0.8
```

### LLM Quality Filtering (UltraFineWeb)

Download the [UltraFineWeb Classifier](https://huggingface.co/openbmb/Ultra-FineWeb-classifier)
to `data/ultra_fineweb_en.bin` (~1.9 GB), then filter a JSONL dataset:

```bash
./target/release/ft-filter filter \
  --model data/ultra_fineweb_en.bin \
  --input data/alice_chunks.jsonl \
  --output data/alice_hq.jsonl \
  --format jsonl \
  --json-key text \
  --label "__label__pos" \
  --threshold 0.5
```

### Performance comparison

```bash
# Default: language-ID model, local FineWeb corpus, 1 warmup + 3 measured runs per engine
pixi run bench

# Short smoke test, or a longer run with machine-readable results
pixi run bench --limit 500 --warmups 0 --runs 1
pixi run bench --runs 5 --report /tmp/ft-filter-benchmark.json

# Compare the quality classifier instead
pixi run bench --model data/ultra_fineweb_en.bin --label __label__pos
```

Use `pixi run bench --help` for input, JSON key, threshold, label, and optional
Linux CPU-affinity (`--cpu`) settings. Models and input must already exist;
benchmarking never downloads data. The release build runs before measurements.

The benchmark compares **streaming filtering pipelines**, Rust versus the
official fastText Python binding installed by Pixi. An untimed streaming pass
creates the same normalized JSONL input for both (only the text field, embedded
LF replaced by spaces). Both use the same target label and threshold, write
retained records to `/dev/null`, and must agree on processed bytes, records,
and retained counts. The reference calls `predict(k=-1)` to retrieve the target
score; Rust evaluates the requested label directly. Python JSON parsing and
binding overhead are part of the comparison, so this is **not an isolated C++
versus Rust kernel benchmark**. Run the parity suites separately to verify
numerical agreement.

Each trial uses a fresh process; engine order alternates, warmups are
discarded, and filesystem caches are not cleared. Reports include:

- Processing median/min/max and coefficient of variation
- Model-load and end-to-end wall time, total worker CPU time, and peak RSS
- Records/s, ASCII lexical tokens/s, text MiB/s, retained counts, and speedup
- Optional JSON with every trial, workload sizes, and environment metadata

Processing includes parsing, inference, filtering, and buffered output, but
excludes model loading and runtime startup. End-to-end includes
startup/imports/teardown; Rust end-to-end also includes a small Python
launcher. CPU/RSS cover the entire engine process (including model load and
Python imports for the reference). Tokens are counted once outside timing using
fastText's ASCII separators; they are not subword features and exclude
synthetic EOS. Temporary input uses disk space proportional to the corpus,
while preparation and inference are streaming. There are no per-record timers
or memory-sampling threads. Timings are informative, not pass/fail performance
gates; CPU frequency and background load affect results.

## Adding New Classifiers

Any standard supervised FastText `.bin` model (such as DCLM heuristics, CCNet,
or custom domain classifiers) works out of the box:

1. **Obtain the `.bin` file:** Download the uncompressed model checkpoint
   (avoid `.ftz` quantized files).
2. **Inspect labels:** Extract the label namespace from the model header:

```bash
python3 -c "
with open('data/your_model.bin', 'rb') as f:
    import re
    print(set(re.findall(rb'__label__\w+', f.read(10 * 1024 * 1024))))
"
```

3. **Run `ft-filter`:** Target the desired output label (e.g., `__label__hq`,
   `__label__pos`, or `__label__1`) with your threshold.
