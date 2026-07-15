"""Content-addressed episodic and semantic memory with deterministic retrieval."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class MemoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryRecord:
    hash: str
    kind: str
    content: str
    metadata: dict[str, Any]
    timeline_node_id: str | None
    salience: float
    created_ns: int
    occurrence_id: int


@dataclass(frozen=True)
class SearchResult:
    record: MemoryRecord
    similarity: float


class ByteNGramEncoder:
    """Transparent, untrained hashed byte n-gram encoder.

    This is a deterministic lexical-semantic baseline, not a pretrained
    embedding model. Similarity comes from shared byte n-grams.
    """

    version = "byte-ngram-v1"

    def __init__(self, dimensions: int = 384, min_n: int = 1, max_n: int = 4) -> None:
        if dimensions < 32:
            raise MemoryError("dimensions must be at least 32")
        if not 1 <= min_n <= max_n:
            raise MemoryError("require 1 <= min_n <= max_n")
        self.dimensions = int(dimensions)
        self.min_n = int(min_n)
        self.max_n = int(max_n)

    def encode(self, text: str) -> np.ndarray:
        data = text.encode("utf-8", errors="replace")
        vector = np.zeros(self.dimensions, dtype=np.float64)
        if not data:
            return vector
        padded = b"\x02" + data + b"\x03"
        for size in range(self.min_n, self.max_n + 1):
            for start in range(max(0, len(padded) - size + 1)):
                digest = hashlib.blake2b(padded[start : start + size], digest_size=8).digest()
                number = int.from_bytes(digest, "little")
                index = number % self.dimensions
                sign = 1.0 if (number >> 63) == 0 else -1.0
                vector[index] += sign / size
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class ContentAddressedMemory:
    """SQLite-backed immutable memory objects keyed by SHA-256 content."""

    VALID_KINDS = frozenset({"episodic", "semantic"})

    def __init__(self, path: str | Path, encoder: ByteNGramEncoder | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder or ByteNGramEncoder()
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
                CREATE TABLE IF NOT EXISTS memories (
                    hash TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK(kind IN ('episodic', 'semantic')),
                    content TEXT NOT NULL,
                    created_ns INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
                CREATE TABLE IF NOT EXISTS memory_occurrences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash TEXT NOT NULL REFERENCES memories(hash) ON DELETE CASCADE,
                    metadata_json TEXT NOT NULL,
                    timeline_node_id TEXT,
                    salience REAL NOT NULL CHECK(salience >= 0 AND salience <= 1),
                    observed_ns INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_occurrences_hash_time
                    ON memory_occurrences(hash, observed_ns DESC);
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    hash TEXT PRIMARY KEY REFERENCES memories(hash) ON DELETE CASCADE,
                    encoder_version TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL
                );
                """
            )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ContentAddressedMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def remember(
        self,
        content: str,
        *,
        kind: str,
        metadata: Mapping[str, Any] | None = None,
        timeline_node_id: str | None = None,
        salience: float = 0.5,
    ) -> MemoryRecord:
        content = content.strip()
        if not content:
            raise MemoryError("memory content cannot be empty")
        if kind not in self.VALID_KINDS:
            raise MemoryError(f"kind must be one of {sorted(self.VALID_KINDS)}")
        if not 0 <= salience <= 1:
            raise MemoryError("salience must be within [0, 1]")
        metadata_dict = dict(metadata or {})
        try:
            metadata_json = json.dumps(
                metadata_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        except (TypeError, ValueError) as exc:
            raise MemoryError("metadata must be JSON-serializable") from exc
        material = f"{kind}\0{content}".encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()
        vector = self.encoder.encode(content).astype("<f8", copy=False)
        created_ns = time.time_ns()
        with self._lock, self._db:
            self._db.execute(
                """INSERT OR IGNORE INTO memories
                   (hash, kind, content, created_ns) VALUES (?, ?, ?, ?)""",
                (digest, kind, content, created_ns),
            )
            self._db.execute(
                """INSERT INTO memory_occurrences
                   (hash, metadata_json, timeline_node_id, salience, observed_ns)
                   VALUES (?, ?, ?, ?, ?)""",
                (digest, metadata_json, timeline_node_id, float(salience), created_ns),
            )
            self._db.execute(
                """INSERT OR IGNORE INTO memory_vectors
                   (hash, encoder_version, dimensions, vector) VALUES (?, ?, ?, ?)""",
                (digest, self.encoder.version, self.encoder.dimensions, vector.tobytes()),
            )
        return self.get(digest)

    def remember_episode(self, content: str, **kwargs: Any) -> MemoryRecord:
        return self.remember(content, kind="episodic", **kwargs)

    def remember_semantic(self, content: str, **kwargs: Any) -> MemoryRecord:
        return self.remember(content, kind="semantic", **kwargs)

    def get(self, digest: str) -> MemoryRecord:
        row = self._db.execute(
            """SELECT m.hash, m.kind, m.content, o.id AS occurrence_id,
                      o.metadata_json, o.timeline_node_id, o.salience,
                      o.observed_ns AS created_ns
               FROM memories m JOIN memory_occurrences o ON o.id = (
                   SELECT latest.id FROM memory_occurrences latest
                   WHERE latest.hash = m.hash
                   ORDER BY latest.observed_ns DESC, latest.id DESC LIMIT 1
               ) WHERE m.hash = ?""",
            (digest,),
        ).fetchone()
        if row is None:
            raise MemoryError(f"unknown memory hash: {digest}")
        return self._record(row)

    def search(
        self, query: str, *, kind: str | None = None, limit: int = 5, min_similarity: float = -1.0
    ) -> list[SearchResult]:
        if kind is not None and kind not in self.VALID_KINDS:
            raise MemoryError(f"kind must be one of {sorted(self.VALID_KINDS)}")
        if limit < 1:
            raise MemoryError("limit must be positive")
        query_vector = self.encoder.encode(query)
        sql = (
            "SELECT m.hash, m.kind, m.content, o.id AS occurrence_id, "
            "o.metadata_json, o.timeline_node_id, o.salience, o.observed_ns AS created_ns, "
            "v.dimensions, v.vector FROM memories m "
            "JOIN memory_occurrences o ON o.id = ("
            "SELECT latest.id FROM memory_occurrences latest WHERE latest.hash = m.hash "
            "ORDER BY latest.observed_ns DESC, latest.id DESC LIMIT 1) "
            "JOIN memory_vectors v ON v.hash = m.hash"
        )
        params: tuple[Any, ...] = ()
        if kind:
            sql += " WHERE m.kind = ?"
            params = (kind,)
        results: list[SearchResult] = []
        for row in self._db.execute(sql, params):
            if int(row["dimensions"]) != self.encoder.dimensions:
                continue
            vector = np.frombuffer(row["vector"], dtype="<f8")
            similarity = float(np.dot(query_vector, vector))
            if similarity >= min_similarity:
                results.append(SearchResult(self._record(row), similarity))
        results.sort(key=lambda item: (-item.similarity, -item.record.salience, item.record.hash))
        return results[:limit]

    def recent(self, limit: int = 20, kind: str | None = None) -> list[MemoryRecord]:
        if limit < 1:
            raise MemoryError("limit must be positive")
        if kind is not None and kind not in self.VALID_KINDS:
            raise MemoryError(f"kind must be one of {sorted(self.VALID_KINDS)}")
        if kind:
            rows = self._db.execute(
                """SELECT m.hash, m.kind, m.content, o.id AS occurrence_id,
                          o.metadata_json, o.timeline_node_id, o.salience,
                          o.observed_ns AS created_ns
                   FROM memory_occurrences o JOIN memories m ON m.hash = o.hash
                   WHERE m.kind = ? ORDER BY o.observed_ns DESC, o.id DESC LIMIT ?""",
                (kind, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                """SELECT m.hash, m.kind, m.content, o.id AS occurrence_id,
                          o.metadata_json, o.timeline_node_id, o.salience,
                          o.observed_ns AS created_ns
                   FROM memory_occurrences o JOIN memories m ON m.hash = o.hash
                   ORDER BY o.observed_ns DESC, o.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def count(self, kind: str | None = None) -> int:
        if kind:
            row = self._db.execute("SELECT COUNT(*) AS n FROM memories WHERE kind = ?", (kind,)).fetchone()
        else:
            row = self._db.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        return int(row["n"])

    def occurrence_count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) AS n FROM memory_occurrences").fetchone()
        return int(row["n"])

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            hash=str(row["hash"]),
            kind=str(row["kind"]),
            content=str(row["content"]),
            metadata=json.loads(row["metadata_json"]),
            timeline_node_id=str(row["timeline_node_id"]) if row["timeline_node_id"] else None,
            salience=float(row["salience"]),
            created_ns=int(row["created_ns"]),
            occurrence_id=int(row["occurrence_id"]),
        )
