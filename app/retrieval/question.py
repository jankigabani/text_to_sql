"""Question expansion — run before retrieval (stage 2, step 4 of the walkthrough).

A raw question is a poor search query: the words that matter for finding a *column*
("city", "status") and the words that matter for finding a *value* ("Thunder Bay",
"2013") play different roles. So we split the question into:

  keywords  — content words, used to search the schema
  literals  — quoted strings, proper nouns, numbers and dates, used for value lookup
  intent    — aggregation / ranking / filtering hints that shape the prompt

This is deliberately rule-based: no LLM call, so it costs nothing and is deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "by", "with", "and", "or", "is", "are",
    "was", "were", "be", "been", "what", "which", "who", "whom", "whose", "how", "many",
    "much", "show", "list", "give", "me", "all", "find", "get", "tell", "please", "do",
    "does", "did", "that", "this", "these", "those", "from", "at", "as", "it", "its",
    "there", "their", "we", "i", "you", "can", "could", "would", "should", "have", "has",
}

AGG_HINTS = {
    "count": ("how many", "number of", "count"),
    "sum": ("total", "sum", "altogether"),
    "avg": ("average", "mean", "typical"),
    "max": ("highest", "largest", "most", "maximum", "top", "best"),
    "min": ("lowest", "smallest", "least", "minimum", "worst"),
}
RANKING = ("top", "highest", "lowest", "most", "least", "best", "worst", "rank", "first")
TIME = ("year", "month", "quarter", "day", "week", "date", "last", "since", "between", "during")

_QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,60})[\"'“”‘’]")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_PROPER = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


@dataclass
class ExpandedQuestion:
    raw: str
    keywords: list[str] = field(default_factory=list)
    literals: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    wants_ranking: bool = False
    mentions_time: bool = False

    @property
    def search_text(self) -> str:
        """What we embed: the question plus its own keywords, which up-weights them."""
        return f"{self.raw} || {' '.join(self.keywords)}"


def expand(question: str) -> ExpandedQuestion:
    q = question.strip()
    low = q.lower()

    literals = [m.group(1) for m in _QUOTED.finditer(q)]
    literals += [m.group(1) for m in _PROPER.finditer(q) if m.start() > 0]
    literals += _NUMBER.findall(q)
    seen: set[str] = set()
    literals = [x for x in literals if not (x.lower() in seen or seen.add(x.lower()))][:12]

    keywords = [w.lower() for w in _WORD.findall(q) if w.lower() not in STOPWORDS and len(w) > 2]
    keywords = list(dict.fromkeys(keywords))[:20]

    aggs = [name for name, hints in AGG_HINTS.items() if any(h in low for h in hints)]

    return ExpandedQuestion(
        raw=q,
        keywords=keywords,
        literals=literals,
        aggregations=aggs,
        wants_ranking=any(w in low for w in RANKING),
        mentions_time=any(w in low for w in TIME),
    )
