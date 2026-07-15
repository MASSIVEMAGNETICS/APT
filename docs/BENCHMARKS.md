# Benchmark protocol

The harness compares three repository-native byte predictors: APT multi-scale
trace, Elman RNN and causal transformer. None uses downloaded weights.

## Reproducibility

- Input is raw bytes; there is no tokenizer advantage.
- The first 80% is training data and the final 20% is validation data.
- Every model receives the same bounded corpus and epoch count.
- Initialization uses the reported seed.
- The metric is next-byte cross-entropy in bits per byte.
- Parameter count, runtime and throughput are reported beside loss.
- Greedy generation uses temperature zero and is emitted as UTF-8 plus exact hex.

## Interpretation

Lower validation bits per byte means the model assigned more probability to the
observed next bytes. It does not measure reasoning, truth, safety, consciousness,
long-context recall or general intelligence.

The architectures are not parameter-matched and their default learning rates
differ. Small-corpus performance is sensitive to repetition and split ordering.
Results are useful for regression detection and engineering comparison, not for
claims of state-of-the-art performance.

For a meaningful experiment, provide a versioned corpus, run multiple seeds,
report mean and dispersion, match parameter or compute budgets, preserve raw JSON
outputs, and evaluate on a held-out corpus not used for architecture decisions.

