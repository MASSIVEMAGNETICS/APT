"""Integrated APT cognitive substrate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .memory import ContentAddressedMemory, MemoryRecord, SearchResult
from .metrics import CognitiveMetrics, MetricReport
from .models import APTByteModel
from .simulator import Hypothesis, ScoredHypothesis, SimulatorBank
from .timeline import TimelineDAG, TimelineNode


class CognitiveOrganism:
    """Coordinates deterministic state, memory, measurement, simulation, and learning."""

    def __init__(self, home: str | Path) -> None:
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.timeline = TimelineDAG(self.home / "timeline.sqlite3")
        self.memory = ContentAddressedMemory(self.home / "memory.sqlite3")
        self.metrics = CognitiveMetrics(self.memory.encoder)
        self.simulators = SimulatorBank(self.metrics)
        self.model_path = self.home / "apt-byte-model.npz"
        self.model = APTByteModel.load(self.model_path) if self.model_path.exists() else APTByteModel()
        if not self.timeline.has_branch("main"):
            self.timeline.genesis(
                {
                    "event": "genesis",
                    "schema": 1,
                    "model_type": self.model.model_type,
                    "claim": "local deterministic research substrate",
                }
            )

    def close(self) -> None:
        self.memory.close()
        self.timeline.close()

    def __enter__(self) -> "CognitiveOrganism":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def observe(
        self,
        text: str,
        *,
        branch: str = "main",
        metadata: Mapping[str, Any] | None = None,
        salience: float = 0.5,
        semantic: bool = False,
    ) -> tuple[TimelineNode, MemoryRecord, MetricReport]:
        text = text.strip()
        if not text:
            raise ValueError("observation cannot be empty")
        context = [record.content for record in self.memory.recent(limit=20)]
        report = self.metrics.measure(text, context)
        node = self.timeline.commit(
            {
                "event": "observation",
                "text": text,
                "metadata": dict(metadata or {}),
                "metrics": report.as_dict(),
            },
            branch,
        )
        kwargs = {
            "metadata": metadata,
            "timeline_node_id": node.id,
            "salience": salience,
        }
        record = (
            self.memory.remember_semantic(text, **kwargs)
            if semantic
            else self.memory.remember_episode(text, **kwargs)
        )
        return node, record, report

    def recall(
        self, query: str, *, kind: str | None = None, limit: int = 5
    ) -> list[SearchResult]:
        return self.memory.search(query, kind=kind, limit=limit)

    def consider(
        self,
        prompt: str,
        hypotheses: Sequence[Hypothesis],
        *,
        branch: str = "main",
        context: Sequence[str] = (),
    ) -> tuple[TimelineNode, list[ScoredHypothesis]]:
        ranked = self.simulators.evaluate(prompt, hypotheses, context=context)
        node = self.timeline.commit(
            {
                "event": "simulation",
                "prompt": prompt,
                "ranked_hypotheses": [item.as_dict() for item in ranked],
            },
            branch,
        )
        winner = ranked[0]
        self.memory.remember_semantic(
            winner.hypothesis.prediction,
            metadata={
                "source": "simulator-bank",
                "prompt": prompt,
                "utility": winner.utility,
                "comparison_weight": winner.normalized_probability,
            },
            timeline_node_id=node.id,
            salience=min(1.0, winner.utility),
        )
        return node, ranked

    def train(
        self,
        corpus: bytes,
        *,
        epochs: int = 1,
        learning_rate: float = 0.01,
        branch: str = "main",
    ) -> list[float]:
        before = self.model.loss(corpus)
        history = self.model.train(corpus, epochs=epochs, learning_rate=learning_rate)
        after = self.model.loss(corpus)
        self.model.save(self.model_path)
        self.timeline.commit(
            {
                "event": "model_training",
                "model_type": self.model.model_type,
                "corpus_bytes": len(corpus),
                "epochs": epochs,
                "learning_rate": learning_rate,
                "bits_per_byte_before": before,
                "bits_per_byte_after": after,
                "epoch_training_loss": history,
            },
            branch,
        )
        return history

    def generate(
        self, prompt: str, *, max_new_bytes: int = 128, temperature: float = 0.8, seed: int = 0
    ) -> str:
        result = self.model.generate(
            prompt.encode("utf-8"),
            max_new_bytes=max_new_bytes,
            temperature=temperature,
            seed=seed,
        )
        return result.decode("utf-8", errors="replace")

    def branch(self, name: str, *, source: str = "main") -> TimelineNode:
        return self.timeline.fork(source, name)

    def rewind(self, *, branch: str = "main", steps: int = 1) -> TimelineNode:
        return self.timeline.rewind(branch, steps)

    def replay(self, branch: str = "main") -> list[TimelineNode]:
        return self.timeline.replay(branch=branch)

    def status(self) -> dict[str, Any]:
        return {
            "home": str(self.home.resolve()),
            "branches": self.timeline.branches(),
            "memory_count": self.memory.count(),
            "memory_occurrence_count": self.memory.occurrence_count(),
            "episodic_count": self.memory.count("episodic"),
            "semantic_count": self.memory.count("semantic"),
            "model_type": self.model.model_type,
            "model_parameters": self.model.parameter_count(),
            "model_trained": self.model_path.exists(),
        }
