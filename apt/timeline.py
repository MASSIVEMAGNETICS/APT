"""Persistent SQLite timeline DAG with branching, rewind, checkout, and replay."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


class TimelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimelineNode:
    id: str
    parent_id: str | None
    branch: str
    sequence: int
    state: dict[str, Any]
    state_hash: str
    created_ns: int


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TimelineError("timeline state must be JSON-serializable") from exc


class TimelineDAG:
    """Append-only node storage with movable per-branch heads.

    Rewind and checkout move a branch head but never delete history. A later
    commit creates a new child, preserving both futures in the DAG.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA synchronous = FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS timeline_nodes (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT REFERENCES timeline_nodes(id),
                    branch TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    created_ns INTEGER NOT NULL,
                    UNIQUE(branch, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_timeline_parent ON timeline_nodes(parent_id);
                CREATE INDEX IF NOT EXISTS idx_timeline_state_hash ON timeline_nodes(state_hash);
                CREATE TABLE IF NOT EXISTS timeline_heads (
                    branch TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL REFERENCES timeline_nodes(id)
                );
                """
            )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "TimelineDAG":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _validate_branch(branch: str) -> str:
        branch = branch.strip()
        if not branch or len(branch) > 128:
            raise TimelineError("branch name must contain 1-128 characters")
        if any(char in branch for char in "\x00\n\r"):
            raise TimelineError("branch name contains an invalid character")
        return branch

    def genesis(self, state: Mapping[str, Any], branch: str = "main") -> TimelineNode:
        branch = self._validate_branch(branch)
        if self.has_branch(branch):
            raise TimelineError(f"branch already exists: {branch}")
        return self._insert(state, branch, parent_id=None)

    def commit(self, state: Mapping[str, Any], branch: str = "main") -> TimelineNode:
        branch = self._validate_branch(branch)
        parent = self.head(branch)
        return self._insert(state, branch, parent_id=parent.id)

    def _insert(
        self, state: Mapping[str, Any], branch: str, parent_id: str | None
    ) -> TimelineNode:
        state_json = _canonical_json(state)
        state_hash = hashlib.sha256(state_json.encode("utf-8")).hexdigest()
        created_ns = time.time_ns()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 AS next FROM timeline_nodes WHERE branch = ?",
                (branch,),
            ).fetchone()
            sequence = int(row["next"])
            material = f"{parent_id or ''}\0{branch}\0{sequence}\0{state_hash}".encode("utf-8")
            node_id = hashlib.sha256(material).hexdigest()
            self._db.execute(
                """INSERT INTO timeline_nodes
                   (id, parent_id, branch, sequence, state_json, state_hash, created_ns)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (node_id, parent_id, branch, sequence, state_json, state_hash, created_ns),
            )
            self._db.execute(
                """INSERT INTO timeline_heads(branch, node_id) VALUES (?, ?)
                   ON CONFLICT(branch) DO UPDATE SET node_id = excluded.node_id""",
                (branch, node_id),
            )
        return TimelineNode(node_id, parent_id, branch, sequence, dict(state), state_hash, created_ns)

    def has_branch(self, branch: str) -> bool:
        row = self._db.execute("SELECT 1 FROM timeline_heads WHERE branch = ?", (branch,)).fetchone()
        return row is not None

    def head(self, branch: str = "main") -> TimelineNode:
        row = self._db.execute(
            """SELECT n.* FROM timeline_heads h
               JOIN timeline_nodes n ON n.id = h.node_id WHERE h.branch = ?""",
            (branch,),
        ).fetchone()
        if row is None:
            raise TimelineError(f"unknown branch: {branch}")
        return self._to_node(row)

    def get(self, node_id: str) -> TimelineNode:
        row = self._db.execute("SELECT * FROM timeline_nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise TimelineError(f"unknown timeline node: {node_id}")
        return self._to_node(row)

    def fork(
        self, source_branch: str, new_branch: str, at_node_id: str | None = None
    ) -> TimelineNode:
        new_branch = self._validate_branch(new_branch)
        if self.has_branch(new_branch):
            raise TimelineError(f"branch already exists: {new_branch}")
        node = self.get(at_node_id) if at_node_id else self.head(source_branch)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO timeline_heads(branch, node_id) VALUES (?, ?)",
                (new_branch, node.id),
            )
        return node

    def checkout(self, branch: str, node_id: str) -> TimelineNode:
        branch = self._validate_branch(branch)
        if not self.has_branch(branch):
            raise TimelineError(f"unknown branch: {branch}")
        node = self.get(node_id)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE timeline_heads SET node_id = ? WHERE branch = ?", (node.id, branch)
            )
        return node

    def rewind(self, branch: str = "main", steps: int = 1) -> TimelineNode:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise TimelineError("steps must be a non-negative integer")
        node = self.head(branch)
        for _ in range(steps):
            if node.parent_id is None:
                break
            node = self.get(node.parent_id)
        return self.checkout(branch, node.id)

    def replay(self, *, branch: str | None = None, node_id: str | None = None) -> list[TimelineNode]:
        if (branch is None) == (node_id is None):
            raise TimelineError("provide exactly one of branch or node_id")
        node = self.head(branch) if branch is not None else self.get(str(node_id))
        path: list[TimelineNode] = []
        while True:
            path.append(node)
            if node.parent_id is None:
                break
            node = self.get(node.parent_id)
        path.reverse()
        return path

    def children(self, node_id: str) -> list[TimelineNode]:
        rows = self._db.execute(
            "SELECT * FROM timeline_nodes WHERE parent_id = ? ORDER BY created_ns, id", (node_id,)
        ).fetchall()
        return [self._to_node(row) for row in rows]

    def branches(self) -> dict[str, str]:
        rows = self._db.execute("SELECT branch, node_id FROM timeline_heads ORDER BY branch").fetchall()
        return {str(row["branch"]): str(row["node_id"]) for row in rows}

    def iter_nodes(self) -> Iterator[TimelineNode]:
        rows = self._db.execute("SELECT * FROM timeline_nodes ORDER BY created_ns, id")
        for row in rows:
            yield self._to_node(row)

    @staticmethod
    def _to_node(row: sqlite3.Row) -> TimelineNode:
        return TimelineNode(
            id=str(row["id"]),
            parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
            branch=str(row["branch"]),
            sequence=int(row["sequence"]),
            state=json.loads(row["state_json"]),
            state_hash=str(row["state_hash"]),
            created_ns=int(row["created_ns"]),
        )

