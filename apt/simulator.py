"""Simulator bank for reproducibly scoring competing hypotheses."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .metrics import CognitiveMetrics


class SimulatorError(ValueError):
    pass


def _unit(name: str, value: float) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise SimulatorError(f"{name} must be within [0, 1]")
    return value


@dataclass(frozen=True)
class Hypothesis:
    prediction: str
    evidence_strength: float
    identifier: str = ""
    prior_probability: float = 0.5
    complexity: float = 0.5
    risk: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prediction.strip():
            raise SimulatorError("hypothesis prediction cannot be empty")
        _unit("evidence_strength", self.evidence_strength)
        _unit("prior_probability", self.prior_probability)
        _unit("complexity", self.complexity)
        _unit("risk", self.risk)
        if not self.identifier:
            digest = hashlib.sha256(self.prediction.encode("utf-8")).hexdigest()[:12]
            object.__setattr__(self, "identifier", f"hypothesis-{digest}")


@dataclass(frozen=True)
class ScoredHypothesis:
    hypothesis: Hypothesis
    coherence: float
    novelty: float
    utility: float
    normalized_probability: float
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": asdict(self.hypothesis),
            "coherence": self.coherence,
            "novelty": self.novelty,
            "utility": self.utility,
            "normalized_probability": self.normalized_probability,
            "rank": self.rank,
        }


class SimulatorBank:
    """Ranks candidate futures using an explicit, auditable score.

    Utility = 0.30 evidence + 0.25 coherence + 0.15 prior + 0.10 novelty
              + 0.10 simplicity + 0.10 safety.
    The normalized probability is a temperature-scaled softmax over utilities;
    it is a comparison weight, not a calibrated real-world probability.
    """

    def __init__(self, metrics: CognitiveMetrics | None = None, temperature: float = 0.15) -> None:
        if temperature <= 0:
            raise SimulatorError("temperature must be positive")
        self.metrics = metrics or CognitiveMetrics()
        self.temperature = float(temperature)

    def evaluate(
        self,
        prompt: str,
        hypotheses: Sequence[Hypothesis],
        *,
        context: Sequence[str] = (),
    ) -> list[ScoredHypothesis]:
        if not prompt.strip():
            raise SimulatorError("prompt cannot be empty")
        if len(hypotheses) < 2:
            raise SimulatorError("at least two competing hypotheses are required")
        if len({item.identifier for item in hypotheses}) != len(hypotheses):
            raise SimulatorError("hypothesis identifiers must be unique")
        raw: list[tuple[Hypothesis, float, float, float]] = []
        scoring_context = [prompt, *context]
        for item in hypotheses:
            coherence = self.metrics.coherence(item.prediction, scoring_context)
            novelty = self.metrics.novelty(item.prediction, context)
            utility = (
                0.30 * item.evidence_strength
                + 0.25 * coherence
                + 0.15 * item.prior_probability
                + 0.10 * novelty
                + 0.10 * (1.0 - item.complexity)
                + 0.10 * (1.0 - item.risk)
            )
            raw.append((item, coherence, novelty, utility))
        maximum = max(item[3] for item in raw)
        exponentials = [math.exp((item[3] - maximum) / self.temperature) for item in raw]
        normalizer = sum(exponentials)
        weighted = [
            (item, coherence, novelty, utility, exponential / normalizer)
            for (item, coherence, novelty, utility), exponential in zip(raw, exponentials)
        ]
        weighted.sort(key=lambda item: (-item[3], item[0].identifier))
        return [
            ScoredHypothesis(
                hypothesis=item,
                coherence=coherence,
                novelty=novelty,
                utility=utility,
                normalized_probability=probability,
                rank=rank,
            )
            for rank, (item, coherence, novelty, utility, probability) in enumerate(weighted, 1)
        ]

