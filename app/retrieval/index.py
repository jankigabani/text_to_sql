"""The two retrieval indexes (stage 2), persisted to disk.

Two separate indexes, on purpose: one over table cards, one over column cards. The
retriever queries both and takes the union — the table-first path and the column-first
path recover different things, which is the point of RQ1.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.core.config import get_settings
from app.core.models import Catalog
from app.retrieval.cards import build_cards
from app.retrieval.embedder import HashingEmbedder, cosine, get_embedder

log = logging.getLogger(__name__)


@dataclass
class VectorIndex:
    keys: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def search(self, qvec: Sequence[float], k: int) -> list[tuple[str, float]]:
        scored = [(key, cosine(qvec, vec)) for key, vec in zip(self.keys, self.vectors)]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:k]


@dataclass
class SchemaIndex:
    tables: VectorIndex
    columns: VectorIndex
    provider: str
    schema_version: str
    embedder: object | None = None

    # ── persistence ───────────────────────────────────────────────────────────
    @staticmethod
    def path() -> Path:
        return Path(get_settings().index_dir) / "schema_index.json"

    def save(self) -> Path:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": self.provider,
            "schema_version": self.schema_version,
            "tables": self.tables.__dict__,
            "columns": self.columns.__dict__,
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        log.info("schema index saved: %d tables, %d columns -> %s",
                 len(self.tables.keys), len(self.columns.keys), p)
        return p

    @classmethod
    def load(cls) -> "SchemaIndex | None":
        p = cls.path()
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                tables=VectorIndex(**d["tables"]),
                columns=VectorIndex(**d["columns"]),
                provider=d["provider"],
                schema_version=d.get("schema_version", ""),
            )
        except Exception as exc:
            log.warning("index unreadable (%s); rebuild needed", exc)
            return None

    # ── querying ──────────────────────────────────────────────────────────────
    def embed_query(self, text: str) -> list[float]:
        if self.embedder is None:
            corpus = self.tables.texts + self.columns.texts
            self.embedder = get_embedder(corpus) if self.provider == "ollama" else _fitted(corpus)
        return self.embedder.embed([text])[0]  # type: ignore[union-attr]


def _fitted(corpus: Sequence[str]) -> HashingEmbedder:
    h = HashingEmbedder()
    h.fit(list(corpus))
    return h


def build_index(cat: Catalog) -> SchemaIndex:
    tcards, ccards = build_cards(cat)
    corpus = [t for _, t in tcards] + [t for _, t in ccards]
    embedder = get_embedder(corpus)
    provider = "ollama" if type(embedder).__name__ == "OllamaEmbedder" else "hashing"

    t_vecs = embedder.embed([t for _, t in tcards])
    c_vecs = embedder.embed([t for _, t in ccards])

    idx = SchemaIndex(
        tables=VectorIndex([k for k, _ in tcards], [t for _, t in tcards], t_vecs),
        columns=VectorIndex([k for k, _ in ccards], [t for _, t in ccards], c_vecs),
        provider=provider,
        schema_version=cat.schema_version,
        embedder=embedder,
    )
    return idx


def get_index(cat: Catalog, refresh: bool = False) -> SchemaIndex:
    if not refresh:
        idx = SchemaIndex.load()
        if idx and idx.schema_version == cat.schema_version:
            return idx
        if idx:
            log.info("schema changed since the index was built; rebuilding")
    idx = build_index(cat)
    idx.save()
    return idx
