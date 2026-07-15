"""Reproducible APT/RNN/transformer comparison; no benchmark claims are embedded."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from apt.models import APTByteModel, RNNByteModel, TransformerByteModel


DEFAULT_CORPUS = (
    "APT stores observations in an append-only timeline. "
    "Branches preserve alternate futures without deleting prior states. "
    "Content hashes identify immutable memories. "
    "Measured novelty replaces random novelty. "
    "Benchmarks report losses even when the result is inconvenient.\n"
).encode("utf-8") * 40


def _evaluate(name: str, model: Any, training: bytes, validation: bytes, epochs: int) -> dict[str, Any]:
    initial = model.loss(validation)
    started = time.perf_counter()
    history = model.train(training, epochs=epochs)
    elapsed = time.perf_counter() - started
    final = model.loss(validation)
    generated = model.generate(b"APT ", max_new_bytes=48, temperature=0, seed=0)
    return {
        "model": name,
        "implementation": model.model_type,
        "parameters": model.parameter_count(),
        "training_bytes": len(training),
        "validation_bytes": len(validation),
        "epochs": epochs,
        "initial_validation_bits_per_byte": initial,
        "final_validation_bits_per_byte": final,
        "epoch_training_bits_per_byte": history,
        "training_seconds": elapsed,
        "training_bytes_per_second": (len(training) * epochs) / elapsed if elapsed else None,
        "greedy_sample_hex": generated.hex(),
        "greedy_sample_utf8": generated.decode("utf-8", errors="replace"),
    }


def compare(data: bytes, *, epochs: int = 1, max_bytes: int = 8192, seed: int = 37) -> dict[str, Any]:
    if len(data) < 64:
        raise ValueError("benchmark corpus must contain at least 64 bytes")
    bounded = data[:max_bytes]
    split = max(32, int(len(bounded) * 0.8))
    if len(bounded) - split < 2:
        split = len(bounded) - 2
    training, validation = bounded[:split], bounded[split:]
    models = [
        ("APT multi-scale trace", APTByteModel(decays=(0.0, 0.5, 0.9), seed=seed)),
        ("Elman RNN baseline", RNNByteModel(hidden_size=64, seed=seed)),
        (
            "Causal transformer baseline",
            TransformerByteModel(
                model_size=32, context_length=16, feedforward_size=64, seed=seed
            ),
        ),
    ]
    results = [_evaluate(name, model, training, validation, epochs) for name, model in models]
    return {
        "schema": 1,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "methodology": {
            "split": "first 80% train / final 20% validation",
            "metric": "next-byte cross-entropy in bits per byte; lower is better",
            "initialization": f"all weights locally randomized from seed {seed}",
            "optimization": "Adam with model-default rates: APT 0.01, RNN 0.003, transformer 0.001",
            "caveat": "architectures are not parameter-matched; report parameter counts and do not generalize from this small corpus",
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-bytes", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = args.corpus.read_bytes() if args.corpus else DEFAULT_CORPUS
    result = compare(data, epochs=args.epochs, max_bytes=args.max_bytes, seed=args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
