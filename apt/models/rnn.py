"""Fully trainable Elman RNN byte baseline implemented directly in NumPy."""

from __future__ import annotations

from pathlib import Path

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


class RNNByteModel:
    model_type = "elman-rnn-v1"

    def __init__(self, hidden_size: int = 64, *, seed: int = 37) -> None:
        if hidden_size < 4:
            raise ValueError("hidden_size must be at least 4")
        self.hidden_size = int(hidden_size)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.hidden_size)
        self.embedding = rng.normal(0, scale, (VOCAB_SIZE, self.hidden_size))
        self.recurrent = rng.normal(0, scale, (self.hidden_size, self.hidden_size))
        self.hidden_bias = np.zeros(self.hidden_size)
        self.output = rng.normal(0, scale, (self.hidden_size, VOCAB_SIZE))
        self.output_bias = np.zeros(VOCAB_SIZE)

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "embedding": self.embedding,
            "recurrent": self.recurrent,
            "hidden_bias": self.hidden_bias,
            "output": self.output,
            "output_bias": self.output_bias,
        }

    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.parameters.values())

    def _step(self, token: int, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        next_hidden = np.tanh(self.embedding[token] + hidden @ self.recurrent + self.hidden_bias)
        probabilities = softmax(next_hidden @ self.output + self.output_bias)
        return next_hidden, probabilities

    def train(
        self,
        data: bytes,
        *,
        epochs: int = 1,
        learning_rate: float = 0.003,
        sequence_length: int = 32,
    ) -> list[float]:
        require_training_data(data)
        if epochs < 1 or sequence_length < 1:
            raise ValueError("epochs and sequence_length must be positive")
        optimizer = Adam(self.parameters, learning_rate)
        history: list[float] = []
        for _ in range(epochs):
            hidden = np.zeros(self.hidden_size)
            total_bits = 0.0
            token_count = 0
            for start in range(0, len(data) - 1, sequence_length):
                x = data[start : min(len(data) - 1, start + sequence_length)]
                y = data[start + 1 : start + 1 + len(x)]
                states = [hidden.copy()]
                probabilities: list[np.ndarray] = []
                for token in x:
                    hidden, distribution = self._step(token, hidden)
                    states.append(hidden.copy())
                    probabilities.append(distribution)
                gradients = {
                    name: np.zeros_like(value) for name, value in self.parameters.items()
                }
                next_gradient = np.zeros(self.hidden_size)
                for position in range(len(x) - 1, -1, -1):
                    target = y[position]
                    distribution = probabilities[position]
                    total_bits += cross_entropy_bits(distribution[target])
                    token_count += 1
                    delta = distribution.copy()
                    delta[target] -= 1.0
                    gradients["output"] += np.outer(states[position + 1], delta)
                    gradients["output_bias"] += delta
                    hidden_gradient = delta @ self.output.T + next_gradient
                    raw_gradient = hidden_gradient * (1.0 - states[position + 1] ** 2)
                    gradients["embedding"][x[position]] += raw_gradient
                    gradients["recurrent"] += np.outer(states[position], raw_gradient)
                    gradients["hidden_bias"] += raw_gradient
                    next_gradient = raw_gradient @ self.recurrent.T
                for gradient in gradients.values():
                    gradient /= max(1, len(x))
                clip_gradients(gradients)
                optimizer.step(gradients)
                hidden = states[-1].copy()
            history.append(total_bits / token_count)
        return history

    def loss(self, data: bytes) -> float:
        require_training_data(data)
        hidden = np.zeros(self.hidden_size)
        total = 0.0
        for current, target in zip(data, data[1:]):
            hidden, distribution = self._step(current, hidden)
            total += cross_entropy_bits(distribution[target])
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
        hidden = np.zeros(self.hidden_size)
        distribution = np.full(VOCAB_SIZE, 1.0 / VOCAB_SIZE)
        for token in prefix:
            hidden, distribution = self._step(token, hidden)
        output = bytearray(prompt)
        rng = np.random.default_rng(seed)
        for _ in range(max_new_bytes):
            token = sample_token(distribution, rng, temperature)
            output.append(token)
            hidden, distribution = self._step(token, hidden)
        return bytes(output)

    def save(self, path: str | Path) -> None:
        save_archive(
            path,
            {"model_type": self.model_type, "seed": self.seed, "hidden_size": self.hidden_size},
            self.parameters,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RNNByteModel":
        metadata, arrays = load_archive(path)
        if metadata.get("model_type") != cls.model_type:
            raise ValueError("archive is not an RNN byte model")
        model = cls(int(metadata["hidden_size"]), seed=int(metadata["seed"]))
        for name, parameter in model.parameters.items():
            if arrays[name].shape != parameter.shape:
                raise ValueError(f"incompatible parameter shape for {name}")
            parameter[...] = arrays[name]
        return model

