"""Reproducible novelty, coherence, entropy, and repetition measurements."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .memory import ByteNGramEncoder


@dataclass(frozen=True)
class MetricReport:
    novelty: float
    coherence: float
    byte_entropy: float
    repetition: float
    context_items: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class CognitiveMetrics:
    """Transparent metrics derived from content instead of randomness.

    Novelty is one minus the strongest cosine similarity to prior context.
    Coherence combines semantic alignment with adjacent sentence continuity.
    Entropy is Shannon byte entropy normalized by the eight-bit maximum.
    Repetition is the repeated share of byte trigrams.
    """

    def __init__(self, encoder: ByteNGramEncoder | None = None) -> None:
        self.encoder = encoder or ByteNGramEncoder()

    def novelty(self, text: str, prior: Sequence[str]) -> float:
        if not prior:
            return 1.0
        vector = self.encoder.encode(text)
        similarities = [float(np.dot(vector, self.encoder.encode(item))) for item in prior]
        return _clamp(1.0 - max(0.0, max(similarities)))

    def coherence(self, text: str, context: Sequence[str]) -> float:
        vector = self.encoder.encode(text)
        if context:
            context_scores = [max(0.0, float(np.dot(vector, self.encoder.encode(item)))) for item in context]
            alignment = sum(context_scores) / len(context_scores)
        else:
            alignment = 1.0
        sentences = [piece.strip() for piece in text.replace("!", ".").replace("?", ".").split(".") if piece.strip()]
        if len(sentences) < 2:
            continuity = alignment
        else:
            adjacent = [
                max(0.0, float(np.dot(self.encoder.encode(left), self.encoder.encode(right))))
                for left, right in zip(sentences, sentences[1:])
            ]
            continuity = sum(adjacent) / len(adjacent)
        return _clamp(0.7 * alignment + 0.3 * continuity)

    @staticmethod
    def byte_entropy(text: str) -> float:
        data = text.encode("utf-8", errors="replace")
        if not data:
            return 0.0
        counts = Counter(data)
        entropy = -sum((count / len(data)) * math.log2(count / len(data)) for count in counts.values())
        return _clamp(entropy / 8.0)

    @staticmethod
    def repetition(text: str, n: int = 3) -> float:
        data = text.encode("utf-8", errors="replace")
        if len(data) < n:
            return 0.0
        grams = [data[index : index + n] for index in range(len(data) - n + 1)]
        unique = len(set(grams))
        return _clamp(1.0 - unique / len(grams))

    def measure(self, text: str, context: Sequence[str] = ()) -> MetricReport:
        return MetricReport(
            novelty=self.novelty(text, context),
            coherence=self.coherence(text, context),
            byte_entropy=self.byte_entropy(text),
            repetition=self.repetition(text),
            context_items=len(context),
        )

