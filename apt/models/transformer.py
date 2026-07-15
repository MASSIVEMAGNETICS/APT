"""One-layer causal transformer byte baseline with manual NumPy backpropagation."""

from __future__ import annotations

import math
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


class TransformerByteModel:
    """Minimal causal self-attention baseline, initialized and trained locally."""

    model_type = "causal-transformer-v1"

    def __init__(
        self, model_size: int = 32, context_length: int = 16, feedforward_size: int = 64, *, seed: int = 37
    ) -> None:
        if model_size < 4 or context_length < 2 or feedforward_size < model_size:
            raise ValueError("invalid transformer dimensions")
        self.model_size = int(model_size)
        self.context_length = int(context_length)
        self.feedforward_size = int(feedforward_size)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        scale = 0.4 / math.sqrt(self.model_size)
        self.embedding = rng.normal(0, scale, (VOCAB_SIZE, self.model_size))
        self.position = rng.normal(0, scale, (self.context_length, self.model_size))
        self.query = rng.normal(0, scale, (self.model_size, self.model_size))
        self.key = rng.normal(0, scale, (self.model_size, self.model_size))
        self.value = rng.normal(0, scale, (self.model_size, self.model_size))
        self.attention_output = rng.normal(0, scale, (self.model_size, self.model_size))
        self.feedforward_in = rng.normal(0, scale, (self.model_size, self.feedforward_size))
        self.feedforward_bias = np.zeros(self.feedforward_size)
        self.feedforward_out = rng.normal(0, scale, (self.feedforward_size, self.model_size))
        self.output = rng.normal(0, scale, (self.model_size, VOCAB_SIZE))
        self.output_bias = np.zeros(VOCAB_SIZE)

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "embedding": self.embedding,
            "position": self.position,
            "query": self.query,
            "key": self.key,
            "value": self.value,
            "attention_output": self.attention_output,
            "feedforward_in": self.feedforward_in,
            "feedforward_bias": self.feedforward_bias,
            "feedforward_out": self.feedforward_out,
            "output": self.output,
            "output_bias": self.output_bias,
        }

    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.parameters.values())

    def _forward(self, tokens: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        length = len(tokens)
        x = self.embedding[tokens] + self.position[:length]
        query = x @ self.query
        key = x @ self.key
        value = x @ self.value
        scores = query @ key.T / math.sqrt(self.model_size)
        scores[np.triu_indices(length, 1)] = -1e9
        attention = softmax(scores)
        context = attention @ value
        attention_state = context @ self.attention_output
        residual = x + attention_state
        pre_activation = residual @ self.feedforward_in + self.feedforward_bias
        activation = np.tanh(pre_activation)
        feedforward = activation @ self.feedforward_out
        hidden = residual + feedforward
        logits = hidden @ self.output + self.output_bias
        cache = {
            "tokens": tokens,
            "x": x,
            "query_state": query,
            "key_state": key,
            "value_state": value,
            "attention": attention,
            "context": context,
            "residual": residual,
            "activation": activation,
            "hidden": hidden,
        }
        return logits, cache

    def _loss_and_gradients(
        self, tokens: np.ndarray, targets: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self._forward(tokens)
        probabilities = softmax(logits)
        loss = sum(cross_entropy_bits(probabilities[i, target]) for i, target in enumerate(targets)) / len(tokens)
        delta = probabilities
        delta[np.arange(len(tokens)), targets] -= 1.0
        delta /= len(tokens)
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        gradients["output"] = cache["hidden"].T @ delta
        gradients["output_bias"] = np.sum(delta, axis=0)
        hidden_gradient = delta @ self.output.T

        gradients["feedforward_out"] = cache["activation"].T @ hidden_gradient
        activation_gradient = hidden_gradient @ self.feedforward_out.T
        pre_gradient = activation_gradient * (1.0 - cache["activation"] ** 2)
        gradients["feedforward_in"] = cache["residual"].T @ pre_gradient
        gradients["feedforward_bias"] = np.sum(pre_gradient, axis=0)
        residual_gradient = hidden_gradient + pre_gradient @ self.feedforward_in.T

        gradients["attention_output"] = cache["context"].T @ residual_gradient
        context_gradient = residual_gradient @ self.attention_output.T
        attention_gradient = context_gradient @ cache["value_state"].T
        value_gradient = cache["attention"].T @ context_gradient
        row_dot = np.sum(attention_gradient * cache["attention"], axis=1, keepdims=True)
        score_gradient = cache["attention"] * (attention_gradient - row_dot)
        score_gradient[np.triu_indices(len(tokens), 1)] = 0.0
        scale = math.sqrt(self.model_size)
        query_gradient = score_gradient @ cache["key_state"] / scale
        key_gradient = score_gradient.T @ cache["query_state"] / scale

        gradients["query"] = cache["x"].T @ query_gradient
        gradients["key"] = cache["x"].T @ key_gradient
        gradients["value"] = cache["x"].T @ value_gradient
        x_gradient = (
            residual_gradient
            + query_gradient @ self.query.T
            + key_gradient @ self.key.T
            + value_gradient @ self.value.T
        )
        np.add.at(gradients["embedding"], tokens, x_gradient)
        gradients["position"][: len(tokens)] = x_gradient
        return loss, gradients

    def train(
        self,
        data: bytes,
        *,
        epochs: int = 1,
        learning_rate: float = 0.001,
    ) -> list[float]:
        require_training_data(data)
        if epochs < 1:
            raise ValueError("epochs must be positive")
        optimizer = Adam(self.parameters, learning_rate)
        history: list[float] = []
        for _ in range(epochs):
            total_loss = 0.0
            token_count = 0
            for start in range(0, len(data) - 1, self.context_length):
                stop = min(len(data) - 1, start + self.context_length)
                tokens = np.frombuffer(data[start:stop], dtype=np.uint8).astype(np.int64)
                targets = np.frombuffer(data[start + 1 : stop + 1], dtype=np.uint8).astype(np.int64)
                loss, gradients = self._loss_and_gradients(tokens, targets)
                clip_gradients(gradients)
                optimizer.step(gradients)
                total_loss += loss * len(tokens)
                token_count += len(tokens)
            history.append(total_loss / token_count)
        return history

    def loss(self, data: bytes) -> float:
        require_training_data(data)
        total_loss = 0.0
        token_count = 0
        for start in range(0, len(data) - 1, self.context_length):
            stop = min(len(data) - 1, start + self.context_length)
            tokens = np.frombuffer(data[start:stop], dtype=np.uint8).astype(np.int64)
            targets = np.frombuffer(data[start + 1 : stop + 1], dtype=np.uint8).astype(np.int64)
            probabilities = softmax(self._forward(tokens)[0])
            for position, target in enumerate(targets):
                total_loss += cross_entropy_bits(probabilities[position, target])
            token_count += len(tokens)
        return total_loss / token_count

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
        output = bytearray(prompt or b"\n")
        rng = np.random.default_rng(seed)
        for _ in range(max_new_bytes):
            context = bytes(output[-self.context_length :])
            tokens = np.frombuffer(context, dtype=np.uint8).astype(np.int64)
            probabilities = softmax(self._forward(tokens)[0][-1])
            output.append(sample_token(probabilities, rng, temperature))
        return bytes(output if prompt else output[1:])

    def save(self, path: str | Path) -> None:
        save_archive(
            path,
            {
                "model_type": self.model_type,
                "seed": self.seed,
                "model_size": self.model_size,
                "context_length": self.context_length,
                "feedforward_size": self.feedforward_size,
            },
            self.parameters,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransformerByteModel":
        metadata, arrays = load_archive(path)
        if metadata.get("model_type") != cls.model_type:
            raise ValueError("archive is not a transformer byte model")
        model = cls(
            int(metadata["model_size"]),
            int(metadata["context_length"]),
            int(metadata["feedforward_size"]),
            seed=int(metadata["seed"]),
        )
        for name, parameter in model.parameters.items():
            if arrays[name].shape != parameter.shape:
                raise ValueError(f"incompatible parameter shape for {name}")
            parameter[...] = arrays[name]
        return model

