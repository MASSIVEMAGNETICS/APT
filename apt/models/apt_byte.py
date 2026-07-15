"""Adaptive Predictive Trace byte model with deterministic multi-scale state."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .base import (
    VOCAB_SIZE,
    Adam,
    clip_gradients,
    cross_entropy_bits,
    load_archive,
    require_training_data,
    sample_token,
    save_archive,
    softmax,
)


class APTByteModel:
    """A trainable multi-timescale byte predictor initialized from scratch.

    Each trace is an exponentially decayed distribution over prior bytes. A
    learned softmax readout combines fast and slow traces. The architecture is
    deterministic, inspectable, recurrent in state, and does not use attention
    or pretrained weights.
    """

    model_type = "apt-trace-v1"

    def __init__(
        self, decays: Sequence[float] = (0.0, 0.5, 0.9), *, seed: int = 37
    ) -> None:
        if not decays:
            raise ValueError("at least one trace decay is required")
        decay_array = np.asarray(decays, dtype=np.float64)
        if np.any(decay_array < 0) or np.any(decay_array >= 1):
            raise ValueError("trace decays must be within [0, 1)")
        self.decays = decay_array
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        scale = 0.01 / np.sqrt(len(self.decays))
        self.weights = rng.normal(
            0.0, scale, size=(len(self.decays), VOCAB_SIZE, VOCAB_SIZE)
        ).astype(np.float64)
        self.bias = np.zeros(VOCAB_SIZE, dtype=np.float64)

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights, "bias": self.bias}

    def parameter_count(self) -> int:
        return int(self.weights.size + self.bias.size)

    def _advance(self, traces: np.ndarray, token: int) -> None:
        traces *= self.decays[:, None]
        traces[:, token] += 1.0 - self.decays

    def _probabilities(self, traces: np.ndarray) -> np.ndarray:
        logits = np.einsum("sv,svw->w", traces, self.weights, optimize=True) + self.bias
        return softmax(logits)

    def _feature_matrix(self, data: bytes) -> np.ndarray:
        traces = np.zeros((len(self.decays), VOCAB_SIZE), dtype=np.float64)
        features = np.empty((len(data), traces.size), dtype=np.float64)
        for index, token in enumerate(data):
            self._advance(traces, token)
            features[index] = traces.reshape(-1)
        return features

    def train(
        self,
        data: bytes,
        *,
        epochs: int = 1,
        learning_rate: float = 0.01,
        batch_tokens: int = 64,
    ) -> list[float]:
        require_training_data(data)
        if epochs < 1 or batch_tokens < 1:
            raise ValueError("epochs and batch_tokens must be positive")
        optimizer = Adam(self.parameters, learning_rate)
        history: list[float] = []
        features = self._feature_matrix(data[:-1])
        targets = np.frombuffer(data[1:], dtype=np.uint8).astype(np.int64)
        flat_weights = self.weights.reshape(-1, VOCAB_SIZE)
        for _ in range(epochs):
            total_bits = 0.0
            for start in range(0, len(features), batch_tokens):
                stop = min(len(features), start + batch_tokens)
                batch_features = features[start:stop]
                batch_targets = targets[start:stop]
                probabilities = softmax(batch_features @ flat_weights + self.bias)
                total_bits += sum(
                    cross_entropy_bits(probabilities[row, target])
                    for row, target in enumerate(batch_targets)
                )
                delta = probabilities
                delta[np.arange(len(batch_targets)), batch_targets] -= 1.0
                delta /= len(batch_targets)
                gradients = {
                    "weights": (batch_features.T @ delta).reshape(self.weights.shape),
                    "bias": np.sum(delta, axis=0),
                }
                clip_gradients(gradients)
                optimizer.step(gradients)
            history.append(total_bits / (len(data) - 1))
        return history

    def loss(self, data: bytes) -> float:
        require_training_data(data)
        features = self._feature_matrix(data[:-1])
        targets = np.frombuffer(data[1:], dtype=np.uint8).astype(np.int64)
        probabilities = softmax(features @ self.weights.reshape(-1, VOCAB_SIZE) + self.bias)
        total = sum(
            cross_entropy_bits(probabilities[row, target]) for row, target in enumerate(targets)
        )
        return total / (len(data) - 1)

    def generate(
        self,
        prompt: bytes = b"",
        *,
        max_new_bytes: int = 128,
        temperature: float = 0.8,
        seed: int = 0,
    ) -> bytes:
        if max_new_bytes < 0:
            raise ValueError("max_new_bytes must be non-negative")
        prefix = prompt or b"\n"
        traces = np.zeros((len(self.decays), VOCAB_SIZE), dtype=np.float64)
        for token in prefix:
            self._advance(traces, token)
        output = bytearray(prompt)
        rng = np.random.default_rng(seed)
        for _ in range(max_new_bytes):
            probabilities = self._probabilities(traces)
            token = sample_token(probabilities, rng, temperature)
            output.append(token)
            self._advance(traces, token)
        return bytes(output)

    def save(self, path: str | Path) -> None:
        save_archive(
            path,
            {"model_type": self.model_type, "seed": self.seed, "decays": self.decays.tolist()},
            self.parameters,
        )

    @classmethod
    def load(cls, path: str | Path) -> "APTByteModel":
        metadata, arrays = load_archive(path)
        if metadata.get("model_type") != cls.model_type:
            raise ValueError("archive is not an APT byte model")
        model = cls(metadata["decays"], seed=int(metadata["seed"]))
        if arrays["weights"].shape != model.weights.shape or arrays["bias"].shape != model.bias.shape:
            raise ValueError("model archive has incompatible parameter shapes")
        model.weights[...] = arrays["weights"]
        model.bias[...] = arrays["bias"]
        return model
