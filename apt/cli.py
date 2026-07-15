"""Command-line interface for the APT local cognitive substrate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .simulator import Hypothesis
from .system import CognitiveOrganism


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apt-cognitive",
        description="Deterministic local-first cognitive systems research substrate",
    )
    parser.add_argument("--home", default=".apt", help="persistent APT state directory")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="initialize persistent state")
    commands.add_parser("status", help="show current state summary")

    observe = commands.add_parser("observe", help="measure and persist an observation")
    observe.add_argument("text")
    observe.add_argument("--branch", default="main")
    observe.add_argument("--salience", type=float, default=0.5)
    observe.add_argument("--semantic", action="store_true")

    recall = commands.add_parser("recall", help="search content-addressed memory")
    recall.add_argument("query")
    recall.add_argument("--kind", choices=["episodic", "semantic"])
    recall.add_argument("--limit", type=int, default=5)

    branch = commands.add_parser("branch", help="fork a timeline branch")
    branch.add_argument("name")
    branch.add_argument("--source", default="main")

    rewind = commands.add_parser("rewind", help="move a branch head backward without deleting history")
    rewind.add_argument("--branch", default="main")
    rewind.add_argument("--steps", type=int, default=1)

    replay = commands.add_parser("replay", help="replay states from genesis to a branch head")
    replay.add_argument("--branch", default="main")

    train = commands.add_parser("train", help="train the APT byte model from a local corpus")
    train.add_argument("corpus", type=Path)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--branch", default="main")

    generate = commands.add_parser("generate", help="generate bytes from the locally trained model")
    generate.add_argument("prompt", nargs="?", default="")
    generate.add_argument("--bytes", type=int, default=128, dest="max_new_bytes")
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--seed", type=int, default=0)

    consider = commands.add_parser("consider", help="score competing candidate predictions")
    consider.add_argument("prompt")
    consider.add_argument("--candidate", action="append", required=True)
    consider.add_argument("--evidence", action="append", type=float)
    consider.add_argument("--branch", default="main")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with CognitiveOrganism(args.home) as organism:
        if args.command in {"init", "status"}:
            _json(organism.status())
        elif args.command == "observe":
            node, record, report = organism.observe(
                args.text,
                branch=args.branch,
                salience=args.salience,
                semantic=args.semantic,
            )
            _json({"node_id": node.id, "memory_hash": record.hash, "metrics": report.as_dict()})
        elif args.command == "recall":
            _json(
                [
                    {
                        "hash": item.record.hash,
                        "kind": item.record.kind,
                        "content": item.record.content,
                        "similarity": item.similarity,
                        "salience": item.record.salience,
                    }
                    for item in organism.recall(args.query, kind=args.kind, limit=args.limit)
                ]
            )
        elif args.command == "branch":
            node = organism.branch(args.name, source=args.source)
            _json({"branch": args.name, "head": node.id})
        elif args.command == "rewind":
            node = organism.rewind(branch=args.branch, steps=args.steps)
            _json({"branch": args.branch, "head": node.id, "state": node.state})
        elif args.command == "replay":
            _json(
                [
                    {
                        "id": node.id,
                        "parent_id": node.parent_id,
                        "origin_branch": node.branch,
                        "sequence": node.sequence,
                        "state": node.state,
                    }
                    for node in organism.replay(args.branch)
                ]
            )
        elif args.command == "train":
            corpus = args.corpus.read_bytes()
            history = organism.train(
                corpus,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                branch=args.branch,
            )
            _json({"epoch_training_bits_per_byte": history, "model": str(organism.model_path)})
        elif args.command == "generate":
            print(
                organism.generate(
                    args.prompt,
                    max_new_bytes=args.max_new_bytes,
                    temperature=args.temperature,
                    seed=args.seed,
                )
            )
        elif args.command == "consider":
            evidence = args.evidence or [0.5] * len(args.candidate)
            if len(evidence) != len(args.candidate):
                raise SystemExit("--evidence must be supplied once per --candidate")
            hypotheses = [
                Hypothesis(
                    identifier=f"candidate-{index}", prediction=text, evidence_strength=strength
                )
                for index, (text, strength) in enumerate(zip(args.candidate, evidence), 1)
            ]
            node, ranked = organism.consider(args.prompt, hypotheses, branch=args.branch)
            _json({"node_id": node.id, "ranking": [item.as_dict() for item in ranked]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

