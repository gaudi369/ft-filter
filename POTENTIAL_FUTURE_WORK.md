# Potential Future Work

These are options, not commitments. The primary direction is a portable,
parity-tested corpus inference tool—not a complete replacement for fastText.
The relaxed-parity direction below is a separate experiment, with no measured
speedup yet.

## 1. Priorities for the parity-preserving project

### Platform support and distribution

1. Provide baseline x86_64 builds alongside separately optimized builds, rather
   than requiring x86-64-v3 for every user.
2. Add Linux ARM64 and macOS ARM64 build/test coverage.
3. Add Windows support if there is demand.
4. Publish release binaries with clear OS, architecture, and CPU requirements.
5. Run probability-parity tests on every supported platform, especially around
   floating-point reductions and mathematical-library behavior.

Do not claim parity on platforms we have not verified. Retain the `1e-5`
absolute probability tolerance; exact agreement observed in current tests is
not a universal bitwise guarantee.

Broader portability is likely a smaller investment than a competitive trainer.
Users can already train with official fastText; making their models easy to
deploy on more machines is more valuable than duplicating that capability.

### Multilabel/top-k inference, when a workflow needs it

Distinguish multiclass prediction (one of several competing classes) from
multilabel prediction (several independently applicable labels).

The existing OVA scoring head provides the basis for multilabel inference.
Potential additions:

- All-label and top-k prediction APIs.
- JSONL annotation with labels and scores.
- Global or per-label thresholds.
- Filtering policies such as “any of these labels” and “all of these labels.”
- Shared feature accumulation across requested labels; appropriate tree
  traversal for hierarchical softmax rather than repeated independent paths.

Expose head-specific semantics: softmax scores compete, whereas OVA scores do
not sum to one. Probability parity does not imply probability calibration;
thresholds still need validation on representative data.

Test score parity, ranking, ties, threshold boundaries, and output schemas.
No multilabel throughput improvement has been measured yet.

### Compatibility and robustness

- Broaden reference-model and real-corpus coverage across supported loss heads,
  feature configurations, and platforms.
- Improve diagnostics for unsupported or malformed models, invalid input,
  missing labels, and invalid CLI options.
- Verify record preservation and consistent filtering/annotation behavior.

### Defensible benchmarking

The existing benchmark compares streaming pipelines against the official
fastText Python binding. It includes Python/binding overhead and different
label-selection work; it is not an isolated Rust-versus-C++ kernel comparison.

Add a direct C++ comparison with equivalent input, requested predictions,
thresholds, and output work before claiming an intrinsic inference advantage.
Report processing, model load, end-to-end time, and peak RSS separately. Use
original JSONL as well as normalized inputs, and document CPU/build settings.

### Defer native training

A future trainer could write compatible `.bin` files while retaining inference
parity on the resulting weights. It would not need to reproduce upstream's
training trajectory. Exact training parity—especially with asynchronous
multithreaded updates—is not a useful target.

Training requires dictionary construction, initialization, sampling, gradient
updates, schedules, serialization, and quality/performance evaluation. Pursue
it only for a concrete unmet need, not merely because inference exists.

## 2. Separate direction: relax numerical parity

The strongest initial experiment is a compiled binary classifier for quality
filtering. Keep the learned model and feature semantics, but allow arithmetic
reordering. This is narrower than abandoning fastText compatibility entirely.

### Compile embedding projections into scalar feature weights

For a two-label softmax model, let `v_f` be a feature embedding and `w_1`, `w_0`
be the output rows. Ignoring upstream score smoothing for the moment:

```
h = sum(v_f) / N
p_1 = sigmoid((w_1 - w_0) · h)

s_f = (w_1 - w_0) · v_f       # offline compilation
p_1 = sigmoid(sum(s_f) / N)    # online inference
```

`N` counts feature occurrences, including duplicates. Preserve tokenization,
subwords, word n-grams, pruning, and EOS handling.

For the dim=256 UltraFineWeb model, this replaces 256 weight values and
accumulator additions per feature with one each. Its approximately 2.09 GB
input matrix becomes an approximately 8.2 MB FP32 scalar table, excluding the
dictionary and metadata. Save a compiled artifact so conversion is a one-time
cost rather than startup work.

The transformation is algebraically equivalent in exact arithmetic, but changes
floating-point rounding and reduction order. It must not inherit the current
`1e-5` guarantee without validation. Probability smoothing and empty-feature
behavior require explicit handling.

For filtering alone, ordinary sigmoid thresholds can be converted into logit
thresholds, avoiding the final sigmoid and division. Account separately for
upstream smoothing and boundary thresholds.

Related transformations apply to a selected NS/OVA label and to binary HS,
but each needs its own score transformation. Multiclass HS generally cannot
collapse an entire label path into one scalar: its probability combines several
branch decisions. The 16-dimensional language-ID model has much less scope for
this optimization than the 256-dimensional binary quality model.

### Is this “just a regression model”?

It is a **linear classifier over normalized sparse feature counts**. For
binary softmax, its probability function has the form of **logistic regression**.
Despite its name, logistic regression is a classification method—not ordinary
least-squares regression predicting an unrestricted continuous target.

fastText already has this linear structure for binary classification: the
embedding and output projections compose into a linear score. Compilation
removes that factorization at inference time; it does not retrain the model or
change its task. Upstream score smoothing adds a small distinction from textbook
logistic probabilities.

Training a direct logistic classifier from scratch is a different proposal.
The same functional form does not imply identical fitted weights: feature
preprocessing, optimization, regularization, and the factorized training
parameterization can all affect the result. For multiclass models, the shared
embedding also imposes a rank constraint when its dimension is small enough.

### Appropriate comparison projects

Official fastText remains the primary baseline when compiling an existing
fastText model: it establishes original scores, decisions, quality, and cost.
Our current optimized Rust implementation is the baseline for any claimed
speedup over this project.

If the fork becomes a standalone sparse linear text-classification system,
these are relevant additional comparisons:

- **[Vowpal Wabbit](https://vowpalwabbit.org/):** hashed sparse features, online
  linear learning, logistic loss, and efficient prediction. A strong comparison
  for a streaming hashed-feature classifier.
- **[LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/):** sparse linear
  logistic regression and SVMs. Useful as a mature linear-classification
  baseline; text extraction/vectorization must be supplied separately.
- **[scikit-learn](https://scikit-learn.org/stable/):** `HashingVectorizer` or
  `TfidfVectorizer` with `SGDClassifier(loss="log_loss")` or
  `LogisticRegression`. Useful reproducible quality baselines and prototypes,
  but Python pipeline timing is not an isolated native-kernel comparison.

These are candidates, not benchmarked alternatives. Their default features are
not fastText's features. Separate two questions:

1. **Execution efficiency:** hold feature IDs, weights, and decisions as close
   as possible to constant; include equivalent preprocessing in pipeline tests.
2. **Best classifier for the task:** let each system use appropriate features
   and training, then compare held-out quality, throughput, memory, and training
   cost. Do not present this as numerical parity.

### Is 10× throughput plausible?

For the binary quality model, yes as a hypothesis—not an established result.
A 256× reduction in embedding work does not imply 256× pipeline throughput.
Tokenization, dictionary lookups, n-gram hashing, JSON parsing, and I/O remain.
If those already take more than 10% of processing time, eliminating embedding
work alone cannot deliver 10×. There is no comparable basis yet for expecting
10× on multiclass language ID.

A focused validation sequence:

1. Measure a ceiling probe that retains parsing and feature extraction but skips
   embedding inference. Treat it as a cost probe, not a functioning classifier.
2. Implement scalar compilation for the binary softmax model only.
3. Benchmark against the current optimized Rust binary and official fastText,
   on one core, with original and normalized JSONL.
4. Measure load time, RSS, throughput, score-error distributions, and keep/drop
   disagreement across operational thresholds on a large, varied corpus.
5. Examine near-threshold documents explicitly. Define acceptable disagreement
   before adopting the fork; a small parity sample is insufficient evidence.

Only then consider quantization, early exit, text sampling, or retraining.
Those introduce additional approximation and need separate quality evaluation.
Multithreading can improve aggregate throughput but is not a single-core 10×
inference improvement.
