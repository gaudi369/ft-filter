# ft-filter

A lean, high-throughput, single-threaded CPU inference engine written in Rust for filtering large text corpora using pre-trained FastText binary classifiers (`.bin`).

Built for LLM data curation pipelines (such as FineWeb, UltraFineWeb, and language identification) without Python runtime overhead, PyTorch/ONNX dependencies, or GPU requirements.

---

## Features

- **High Throughput:** >5,000,000 tokens/second single-core inference via memory-mapping (`memmap2`) and SIMD-friendly vector operations.
- **Minimal Footprint:** Stream-processes standard input or files line-by-line without loading corpora into RAM.
- **FastText Parity:** Supports both Hierarchical Softmax (dynamic binary Huffman tree) and standard Softmax/Sigmoid classification heads with byte-level subwords.
- **Zero-Copy Pipeline:** Direct filtering for raw text and JSONL formatted corpora.

---

## Build

```bash
# Build optimized native release binary
RUSTFLAGS="-C target-cpu=native" cargo build --release

```

The compiled binary will be placed at `./target/release/ft-filter`.

---

## CLI Usage

```text
ft-filter [OPTIONS] --model <PATH>

Options:
  -m, --model <PATH>        Path to the FastText .bin file [required]
  -i, --input <PATH>        Input file path (defaults to stdin)
  -o, --output <PATH>       Output file path (defaults to stdout)
  -f, --format <FORMAT>     Input format: 'raw' or 'jsonl' [default: jsonl]
  -k, --json-key <KEY>      JSON field containing the text [default: text]
  -t, --threshold <FLOAT>   Minimum confidence probability [default: 0.5]
  -l, --label <STRING>      Target label to evaluate (e.g. '__label__en', '__label__pos')
  -h, --help                Print help

```

---

## Verification & Quick Start

Run the parity checks in the Pixi environment:

```bash
pixi run test-unit
pixi run test-parity   # requires data/lid.176.bin; also trains tiny temporary reference models
pixi run test-fineweb  # requires data/ultra_fineweb_en.bin; downloads evaluation data if absent
```

Both parity suites enforce an absolute tolerance of `1e-5`. The language-ID
suite covers multilingual text, ASCII versus Unicode whitespace, OOV words,
subwords, word n-grams, EOS, and all supported loss heads. The FineWeb suite
checks 500 reference scores and verifies that the full output preserves records.
JSONL text is scored as one document, equivalent to Python fastText's
`model.predict(text.replace("\n", " "))`; embedded newlines do not terminate it.
Scores retain upstream's `1e-5` smoothing and may slightly exceed 1.0.


### 1. Language Identification (`lid.176.bin`)

Download the official FastText language identifier (~131 MB):

```bash
curl -L -o data/lid.176.bin [https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)

```

Test against raw strings:

```bash
# Should pass through (English)
echo "Alice was beginning to get very tired of sitting by her sister." | \
  ./target/release/ft-filter -m data/lid.176.bin -f raw -l "__label__en" -t 0.8

# Should be dropped (Evaluating Spanish on English text)
echo "Alice was beginning to get very tired of sitting by her sister." | \
  ./target/release/ft-filter -m data/lid.176.bin -f raw -l "__label__es" -t 0.8

```

### 2. LLM Quality Filtering (UltraFineWeb)

Download the [UltraFineWeb Classifier](https://huggingface.co/openbmb/Ultra-FineWeb-classifier) to `data/ultra_fineweb_en.bin` (~1.9 GB), then filter a JSONL dataset:

```bash
./target/release/ft-filter \
  --model data/ultra_fineweb_en.bin \
  --input data/alice_chunks.jsonl \
  --output data/alice_hq.jsonl \
  --format jsonl \
  --json-key text \
  --label "__label__pos" \
  --threshold 0.5

```

---

## Adding New Classifiers

Any standard supervised FastText `.bin` model (such as DCLM heuristics, CCNet, or custom domain classifiers) works out of the box:

1. **Obtain the `.bin` file:** Download the uncompressed model checkpoint (avoid `.ftz` quantized files).
2. **Inspect labels:** Extract the label namespace from the model header:
```bash
python3 -c "
with open('data/your_model.bin', 'rb') as f:
    import re
    print(set(re.findall(rb'__label__\w+', f.read(10 * 1024 * 1024))))
"

```


3. **Run `ft-filter`:** Target the desired output label (e.g., `__label__hq`, `__label__pos`, or `__label__1`) with your threshold.

```


