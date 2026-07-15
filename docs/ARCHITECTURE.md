# Architecture

APT separates durable evidence from learned prediction. A model can be retrained
or replaced without rewriting prior observations, and a timeline branch can be
rewound without deleting alternate futures.

## Integrated event path

1. `CognitiveOrganism.observe` reads recent memory occurrences.
2. `CognitiveMetrics` measures the incoming text against that context.
3. `TimelineDAG.commit` appends the complete observation and metric report.
4. `ContentAddressedMemory.remember_*` stores or reuses the immutable object and
   appends a new occurrence linked to the timeline node.
5. Retrieval recomputes a deterministic query vector and ranks stored vectors by
   cosine similarity, salience, then hash.

The order matters: the durable timeline commit is created before its memory
occurrence so the occurrence can hold a valid node identifier.

## Timeline invariants

- Nodes are append-only.
- Each branch has one movable head.
- A rewind changes only the head.
- A commit after rewind creates another child of the historical node.
- Replay follows parent identifiers back to genesis and reverses that path.
- Node identifiers hash parent, origin branch, branch-local sequence and canonical
  state hash; random IDs are never used.

SQLite uses write-ahead logging, foreign keys and `synchronous=FULL`. These
settings provide strong single-machine crash durability when the filesystem and
SQLite implementation honor synchronization requests.

## Memory invariants

The object hash is `SHA-256(kind + NUL + content)`. Metadata, salience, time and
timeline location belong to occurrences, not immutable objects. Repeating the
same observation therefore produces one object and multiple occurrence records.

The semantic index is explicit and replaceable. Version 1 uses signed hashed byte
n-grams from one through four bytes and L2 normalization. It needs no vocabulary,
network, pretrained model or external service. Its limitations are documented
instead of hidden behind the word “embedding.”

## Learning models

### APT multi-scale predictive trace

For each decay `d` and observed byte `x_t`, a trace evolves as:

```text
trace_t = d * trace_(t-1) + (1 - d) * one_hot(x_t)
```

The concatenated trace bank feeds a learned 256-way softmax readout. Fast traces
capture local transitions; slower traces capture longer byte distributions. The
readout is trained with next-byte cross-entropy and Adam. All weights start from a
seeded local random generator.

### RNN baseline

The Elman baseline trains the byte embedding, recurrent matrix, biases and output
matrix with truncated backpropagation through time. It is not a wrapped framework
model.

### Transformer baseline

The transformer baseline implements one causal self-attention head, causal mask,
residual attention path, tanh feed-forward path and softmax output. Its forward
and backward passes are written directly in NumPy.

## Simulator bank

Candidate hypotheses carry evidence, prior, complexity and risk values in `[0,1]`.
APT measures prediction coherence and novelty, then computes:

```text
utility = 0.30 evidence + 0.25 coherence + 0.15 prior + 0.10 novelty
        + 0.10 (1 - complexity) + 0.10 (1 - risk)
```

A temperature softmax produces relative comparison weights. Calling those values
real-world probabilities would be dishonest; the API names them normalized
comparison probabilities and the documentation preserves the distinction.

