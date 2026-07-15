"""Shared numerical utilities for locally initialized byte models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


VOCAB_SIZE = 256


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def cross_entropy_bits(probability: float) -> float:
    return -math.log2(max(float(probability), 1e-12))


def clip_gradients(gradients: Mapping[str, np.ndarray], max_norm: float = 5.0) -> float:
    norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for gradient in gradients.values():
            gradient *= scale
    return norm


class Adam:
    """Small deterministic Adam optimizer implemented directly in NumPy."""

    def __init__(
        self,
        parameters: Mapping[str, np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.parameters = parameters
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step_count = 0

    def step(self, gradients: Mapping[str, np.ndarray]) -> None:
        self.step_count += 1
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            self.first[name] = self.beta1 * self.first[name] + (1.0 - self.beta1) * gradient
            self.second[name] = self.beta2 * self.second[name] + (1.0 - self.beta2) * gradient * gradient
            first_hat = self.first[name] / correction1
            second_hat = self.second[name] / correction2
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)


def require_training_data(data: bytes) -> None:
    if not isinstance(data, bytes):
        raise TypeError("training data must be bytes")
    if len(data) < 2:
        raise ValueError("training data must contain at least two bytes")


def sample_token(probabilities: np.ndarray, rng: np.random.Generator, temperature: float) -> int:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return int(np.argmax(probabilities))
    adjusted = np.log(np.maximum(probabilities, 1e-12)) / max(temperature, 1e-6)
    distribution = softmax(adjusted)
    return int(rng.choice(VOCAB_SIZE, p=distribution))


def save_archive(path: str | Path, metadata: dict[str, object], arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, metadata=np.array(json.dumps(metadata)), **arrays)


def load_archive(path: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
    return metadata, arrays

