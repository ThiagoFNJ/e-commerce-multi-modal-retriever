"""F2 sentence pipeline — pure functions per system-design.md §2 ingestion hygiene.

Rules (decided): split at sentence punctuation; ≥15-char floor; English-only
(BGE-small is EN); exact-duplicate sentences dedup within a product. Drop rates are
reported by the caller — a review yielding zero sentences contributes no rows
(absence, not a zero vector).
"""

from __future__ import annotations

import re

_SPLIT = re.compile(r"[.!?]+[\s\"')\]]*")
_WS = re.compile(r"\s+")
MIN_CHARS = 15


def split_sentences(text: str) -> list[str]:
    """Punctuation-boundary split, whitespace-normalized, ≥MIN_CHARS floor."""
    if not text:
        return []
    parts = (_WS.sub(" ", p).strip() for p in _SPLIT.split(text))
    return [p for p in parts if len(p) >= MIN_CHARS]


def is_english(sentence: str) -> bool:
    from fast_langdetect import detect

    try:
        result = detect(sentence)  # fasttext lid.176: [{"lang": "en", "score": ...}]
        return result[0]["lang"] == "en" if isinstance(result, list) else result["lang"] == "en"
    except Exception:  # noqa: BLE001 — detector failure must not drop the row silently
        return True


def review_to_rows(text: str, *, lang_filter: bool = True) -> list[str]:
    rows = split_sentences(text)
    if lang_filter:
        rows = [s for s in rows if is_english(s)]
    return rows


def dedup_product(sentences: list[str]) -> list[str]:
    """Exact-duplicate collapse, order-preserving, within one product."""
    seen: set[str] = set()
    out = []
    for s in sentences:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out
