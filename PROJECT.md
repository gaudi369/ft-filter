Project Specification: ft-filter (Single-Threaded CPU FastText Engine)
A lean, single-threaded CPU inference engine designed for local LLM practitioners to filter large text corpora using pre-trained FastText binary classifiers (.bin).
1. Goals and Non-Goals
Goals
 * Native Single-Threaded Throughput: Exceed 100,000 tokens/second on modern consumer x86_64 CPUs (AVX2).
 * Streaming I/O: Classify text line-by-line or from JSONL streams without reading entire datasets into RAM.
 * Low Memory Footprint: Keep RAM overhead under 2 GB (storing only model parameters and small per-document accumulation buffers).
 * Deterministic Output: Produce numerical classification probabilities that match the official C++ FastText output to within floating-point epsilon (\le 10^{-5}).
Non-Goals
 * FP16 / Sub-Byte Quantization: Maintain FP32 weights exclusively to preserve simplicity and avoid conversion latency on consumer x86_64 chips.
 * Multi-Threading / Thread Pools: Keep code strictly single-threaded. Concurrency is delegated to the operating system (e.g., via xargs -P or GNU parallel over separate files).
 * GPU / DirectML / Metal Acceleration: Pure CPU implementation.
 * Dynamic Training Dictionaries: No complex vocabulary tree mutations; treat the loaded model as immutable data.
2. Dependencies & Ecosystem Choices
The implementation should be written in Rust to enforce strict memory safety and avoid external runtime dependencies.
 * memmap2: Memory-mapping model files and input corpora directly into address space with zero disk-to-heap copies.
 * serde + serde_json: Zero-copy or borrowed parsing for JSONL documents (extracting the "text" field).
 * byteorder: Reading little-endian binary headers from FastText .bin checkpoints.
 * clap: CLI flag and argument parsing.
Deliberately Excluded (Avoid Implementing or Adding):
 * tch-rs or onnxruntime: Adds runtime overhead and dynamic library dependencies.
 * External BLAS/LAPACK (OpenBLAS, MKL): Unnecessary; a 256-wide dot product is small enough for straightforward vector intrinsics or auto-vectorized loops.
3. Architecture & Internal Components
A. Model File Parser (model.rs)
The parser reads the official FastText binary layout:
 * Magic Header: Check for FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712298 (0x2F4F16BA).
 * Parameters: Read 32-bit integers: dim (typically 256), ws, epoch, minCount, neg, wordNgrams, loss, model, bucket, minn, maxn, lrUpdateRate, t.
 * Vocabulary Dict: Read vocabulary tokens, word frequencies, and entry types (word vs label). Identify output label IDs.
 * Input Matrix (W_{in}): Float32 array of shape (vocab_size + bucket) x dim.
 * Output Matrix (W_{out}): Float32 array of shape num_labels x dim.
B. Zero-Allocation Tokenizer & Hasher (tokenize.rs)
 * Whitespace & Punctuation Splitting: Scan byte slices (&[u8]) directly without allocating intermediate String instances.
 * FastText FNV-1a Hashing: FastText uses an explicit variant of the 32-bit FNV-1a algorithm for subword n-grams:
   pub fn fasttext_hash(bytes: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in bytes {
        h = (h ^ (b as u32)).wrapping_mul(16777619);
    }
    h
}

 * Subword Generator: For words not in the explicit vocabulary (or in addition to vocabulary words if configured), wrap the word with boundary tokens < and >, slice all character n-grams between minn and maxn, and map each to a bucket index:
   
C. SIMD Accumulator & Linear Head (infer.rs)
 * Accumulator Buffer: Maintain a single reusable stack- or pre-allocated heap-allocated vector acc: [f32; 256] initialized to zeros.
 * Vectorized Addition (AVX2): For each token/subword ID, add the 256-float slice from W_{in} into acc using AVX2 instructions (_mm256_add_ps / _mm256_loadu_ps) or standard 8-element unrolled loops marked with #[inline(always)] to facilitate compiler auto-vectorization.
 * Mean Reduction: Divide all elements in acc by \max(1, N) where N is the total number of valid subwords and tokens in the document.
 * Projection & Activation: Compute the inner product between acc and each row of W_{out}. Apply sigmoid (for multi-label / binary classification) or softmax to determine the confidence score.
D. Stream Pipeline (main.rs)
 * Read from standard input (stdin) or a file path.
 * Iterate line-by-line via BufRead::read_line or iterate records using serde_json::Deserializer::from_reader(stream).into_iter::<Record>().
 * If probability exceeds --threshold (e.g., 0.5), emit the line to standard output (stdout).
4. CLI Interface Specification
ft-filter [OPTIONS] --model <MODEL_PATH>

OPTIONS:
    -m, --model <PATH>          Path to the FastText .bin file [required]
    -i, --input <PATH>          Input file path (defaults to stdin if omitted)
    -o, --output <PATH>         Output file path (defaults to stdout if omitted)
    -f, --format <FORMAT>       Input format: 'raw' (line-by-line) or 'jsonl' [default: jsonl]
    -k, --json-key <KEY>        Key name to extract when using jsonl [default: text]
    -t, --threshold <FLOAT>     Probability threshold to retain document [default: 0.5]
    -l, --label <STRING>        Target label to check (e.g., "__label__hq") [default: auto-detect first]
    -h, --help                  Print help information

5. Verification & Acceptance Criteria
 * Unit Test - Hash Parity: Write a test verifying fasttext_hash against known test vectors from the reference implementation (e.g., "hello", "<the>").
 * Integration Test - Prediction Parity: Given a sample .bin model and 100 test sentences:
   * Run inference with official fasttext predict-prob model.bin -.
   * Run inference with ft-filter.
   * Ensure predicted probabilities match within \pm 0.00001.
 * Benchmark Target: Process an uncompressed text file containing \ge 50\text{MB} of raw text on an AVX2-compatible consumer CPU using a single core at a rate of \ge 80,000 tokens/second.
