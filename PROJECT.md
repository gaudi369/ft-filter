# Project Specification: ft-filter

A lean, single-threaded CPU inference engine that filters large text corpora with
pre-trained FastText binary classifiers (`.bin`). Built for local LLM data
curation (quality filtering, language identification) with no Python, ONNX, or
GPU dependencies at runtime.

## 1. Design Goals

1. **FastText parity.** Classification probabilities match the official C++
   fastText output within `1e-5` (measured: bit-identical on all test suites).
2. **Streaming I/O.** Classify line-by-line or from JSONL streams without loading
   corpora into RAM.
3. **Low memory footprint.** Store only model parameters plus small per-document
   buffers. Peak RSS for the 2 GB UltraFineWeb classifier is ~2.0 GiB.
4. **Single-core throughput.** Targets AVX2-class consumer x86_64 CPUs; measured
   ~2.4M ASCII tokens/s (dim=256) and ~1.9M tokens/s (dim=16, subwords) on one core.
5. **Model generality.** Any supervised FastText `.bin` works: all four loss
   heads (hierarchical softmax, negative sampling, one-vs-all, softmax), any
   embedding dimension, subwords on or off, word n-grams on or off, pruned and
   unpruned dictionaries.

## 2. Intentional Tradeoffs

* **FP32 only.** No FP16 or sub-byte quantization; weights are used exactly as
  saved. (Quantized `.ftz` models are rejected with a clear error.)
* **Strictly single-threaded.** Parallelism is delegated to the OS
  (`xargs -P`, GNU parallel) over separate files.
* **Upstream numerical semantics, not "prettier" math.** Scores reproduce
  upstream's `exp(log(p + 1e-5))` smoothing, the 512-entry sigmoid lookup table
  for NS/OVA, and per-branch root-first log-probability accumulation for
  hierarchical softmax. Consequently scores can slightly exceed 1.0.
* **Exact feature semantics.** ASCII-only token separators, signed-char FNV-1a
  hashing, UTF-8 character subwords including whole words, upstream's wrapping
  uint64 word-ngram hash, and document-level EOS. These reproduce fastText's
  rounding order bit-for-bit rather than approximating it.
* **Vectorization via codegen, not intrinsics.** AVX2 comes from
  `target-cpu=x86-64-v3` in `.cargo/config.toml`; there is no hand-written SIMD
  and no runtime dispatch. Wider registers preserve per-element FP32 add order,
  so parity is unaffected.
* **No mmap for model loading.** FastText matrices sit at byte-unaligned file
  offsets, so any mmap-based loader touches every matrix page and then doubles
  RSS with a heap copy. Matrices are bulk-read from the file in bounded chunks
  instead (see Architecture); `memmap2` was removed when its last use
  disappeared.
* **Training is a stub.** The `train` subcommand exists as a CLI placeholder
  only; models are always loaded, never trained.

## 3. Architecture

```
src/
  model.rs     FastText .bin parser + ordered subword-ID cache
  tokenize.rs  ASCII token splitting, FNV-1a hashing, subword n-grams
  infer.rs     feature accumulation, mean reduction, scoring heads
  main.rs      CLI, stream pipeline, JSONL handling, --stats
```

* **Parser (`model.rs`).** Validates the magic header (`0x2F4F16BA`, version 12)
  and reads the 13 integer parameters plus float64 `t`, the dictionary (token,
  count, entry type; words precede labels), the optional pruning map, and the
  two FP32 matrices. Header/dictionary parse sequentially from a buffered
  reader; matrices are read in 8 MiB chunks into owned buffers, so the file is
  read once and never fully resident. The Huffman tree for hierarchical softmax
  is rebuilt exactly as upstream does from label counts.
* **Tokenizer (`tokenize.rs`).** Splits on fastText's seven ASCII separators
  only (space, LF, CR, tab, VT, FF, NUL) over borrowed slices; no allocations.
  Subword n-grams are generated over the virtual `<word>` with UTF-8 character
  boundaries, excluding only standalone boundary characters, never `</s>`.
* **Subword cache (`model.rs`).** At load time the already-pruned subword IDs
  for the most frequent vocabulary words are stored in original order (bounded
  at 65,536 words / 1,048,576 IDs ≈ 8 MiB). Inference reuses them for known
  words and falls back to on-the-fly generation for everything else. Order,
  duplicates, and pruning semantics are preserved exactly.
* **Inference (`infer.rs`).** One reusable accumulator; features are added in
  upstream order (each word then its subwords, then word n-grams), divided by
  the feature count, and projected through the matching scoring head. Word
  hashes are computed only when the model actually uses word n-grams.
* **Pipeline (`main.rs`).** Reads stdin or `--input` line-by-line (JSONL by
  default: each record's `--json-key` field is scored as one document, embedded
  LF treated as space, matching Python `predict(text.replace("\n", " "))`).
  Records passing `--threshold` are emitted unchanged to stdout or `--output`
  (`--emit-score` annotates them with the probability, `--invert` flips the
  comparison). `filter --stats` prints a JSON line with load/processing time,
  input bytes, record and pass counts to stderr.

## 4. Platform Support Status

| Component | Status |
|---|---|
| Linux x86_64 | **Supported and tested** (development and CI environment; pixi `linux-64`). |
| Binary ISA | Release builds target x86-64-v3 (AVX2 baseline, Haswell 2013+). Older x86_64 CPUs abort with SIGILL; delete `.cargo/config.toml` for a conservative baseline build. |
| macOS / Windows | Untested. The code is platform-neutral Rust; the pixi environment and benchmark tooling assume Linux (`/proc`-free paths except `--stats` VmHWM-free output, `/dev/null`, CPU affinity). |
| Non-x86_64 (ARM64) | Untested. Inference is scalar and portable, but no tuning or verification has been done. |
| `.ftz` quantized models | Not supported (explicit error), by design. |

## 5. Dependencies

Runtime: `clap` (CLI), `byteorder` (LE header fields), `serde` + `serde_json`
(JSONL parsing/annotation). Everything else is `std`. Deliberately excluded:
`tch-rs`/`onnxruntime` (runtime overhead, dynamic libraries), external
BLAS/LAPACK (a 256-wide dot product auto-vectorizes fine), any GPU framework.

## 6. Verification

| Command | What it checks |
|---|---|
| `pixi run test-unit` | Hash test vectors, subword rules, ASCII separator semantics |
| `pixi run test-parity` | Scores vs. official fastText on 5 reference-trained models (all loss heads, mixed n-gram configs, incl. `minn=0`) over 27 edge-case texts × all labels, plus `lid.176.bin` with 127 multilingual texts × 9 labels; tolerance `1e-5` (measured: 0.00e+00) |
| `pixi run test-fineweb` | 500 reference scores on real corpus data + full-output record preservation; tolerance `1e-5` (measured: 0.00e+00) |
| `pixi run bench` | Head-to-head streaming-filter benchmark vs. the official fastText Python binding (see README for methodology and caveats) |

Models and corpora in `data/` are not committed. See README for model downloads;
`pixi run test-fineweb` downloads the evaluation corpus if absent.
