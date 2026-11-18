"""Embeddings for schema retrieval.

Two providers:

* `ollama`  — real sentence embeddings (nomic-embed-text). Used for the reported results.
* `hashing` — a dependency-free hashed bag-of-words + IDF vector. Weaker, but it needs no
  model download, so the project runs (and the tests pass) on a machine with no Ollama.
  Having a deterministic fallback also makes the retrieval tests reproducible.
"""
from __future__ import annotations

import logging
import math
import re
import zlib
from collections import Counter
from typing import Protocol, Sequence

import httpx

from app.core.config import get_settings

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
        self.dim = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for t in texts:
                r = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                )
                r.raise_for_status()
                vec = r.json()["embedding"]
                self.dim = len(vec)
                out.append(_unit(vec))
        return out


class HashingEmbedder:
    """Hashed TF-IDF. Deterministic, offline, no model."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._idf: dict[str, float] = {}

    def fit(self, corpus: Sequence[str]) -> None:
        n = max(len(corpus), 1)
        df = Counter()
        for text in corpus:
            df.update(set(tokenize(text)))
        self._idf = {tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in df.items()}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            counts = Counter(tokenize(text))
            for tok, tf in counts.items():
                w = (1 + math.log(tf)) * self._idf.get(tok, 1.0)
                # crc32, not hash(): str hashing is salted per process, which would make
                # a saved index unusable from a different process.
                vec[zlib.crc32(tok.encode()) % self.dim] += w
            vectors.append(_unit(vec))
        return vectors


def _unit(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def get_embedder(corpus: Sequence[str] | None = None) -> Embedder:
    s = get_settings()
    if s.embed_provider == "ollama":
        emb = OllamaEmbedder(s.ollama_base_url, s.embed_model, s.llm_timeout_s)
        try:  # fail over rather than crash — a missing Ollama should not break the demo
            emb.embed(["probe"])
            return emb
        except Exception as exc:
            log.warning("ollama embeddings unavailable (%s); using hashing fallback", exc)
    h = HashingEmbedder()
    h.fit(list(corpus or ["schema"]))
    return h
