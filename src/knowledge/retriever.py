"""BM25-based retriever for knowledge/ markdown files.

Chunks each file on H2 section headers, builds a BM25 index in-memory, and
retrieves the top-N chunks most relevant to a query composed of series IDs and
free-text keywords.  Used by the composer to pull concept / findings / source
context without loading every section.

Usage
-----
    from src.knowledge.retriever import retrieve_context

    chunks = retrieve_context(
        series_ids=["VIXCLS", "T10Y2Y"],
        keywords=["regime", "volatility"],
        top_n=4,
    )
    combined_text = "\n\n".join(c.text for c in chunks)
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi

_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"

# Files to index, in priority order (earlier files get a score boost).
_FILES = ["concepts.md", "findings.md", "sources.md"]

# Score boost multiplier applied to chunks from concepts.md vs other files.
_CONCEPTS_BOOST = 1.5


@dataclass
class Chunk:
    source: str       # filename
    header: str       # H2 header text
    text: str         # full section text including the header line


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return [t for t in text.split() if len(t) > 1]


def _split_sections(path: Path) -> list[Chunk]:
    """Split a markdown file on ## headers into Chunk objects."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    # Split on H2 headers (## Title), keep the header in the chunk.
    raw_sections = re.split(r"\n(?=## )", text)
    chunks: list[Chunk] = []
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        header = lines[0].lstrip("#").strip() if lines[0].startswith("##") else ""
        chunks.append(Chunk(source=path.name, header=header, text=section))
    return chunks


def _build_index(files: Sequence[str] = _FILES) -> tuple[list[Chunk], BM25Okapi]:
    """Load all knowledge files, chunk by H2, return chunks + BM25 index."""
    all_chunks: list[Chunk] = []
    for fname in files:
        all_chunks.extend(_split_sections(_KNOWLEDGE_DIR / fname))
    if not all_chunks:
        return [], None  # type: ignore[return-value]
    corpus = [_tokenize(c.text) for c in all_chunks]
    return all_chunks, BM25Okapi(corpus)


def retrieve_context(
    series_ids: Sequence[str],
    keywords: Sequence[str] = (),
    top_n: int = 4,
    files: Sequence[str] = _FILES,
    max_chars_per_chunk: int = 1500,
) -> list[Chunk]:
    """Return the top_n most relevant knowledge chunks for the given query.

    Query is built from series_ids + free-text keywords.  Chunks from
    concepts.md receive a 1.5× score boost since they carry institutional
    context the composer needs most.

    Each returned chunk is truncated to max_chars_per_chunk to cap token cost
    on every LLM call. The header line is always preserved; only the body is
    truncated, with a trailing ellipsis when cut.

    Returns an empty list if no knowledge files exist.
    """
    chunks, index = _build_index(files)
    if not chunks:
        return []

    query_terms = _tokenize(" ".join(list(series_ids) + list(keywords)))
    if not query_terms:
        return _truncate_chunks(chunks[:top_n], max_chars_per_chunk)

    raw_scores = index.get_scores(query_terms)

    # Apply per-file boosts and exact series-ID bonus.
    boosted: list[tuple[float, int]] = []
    series_upper = {s.upper() for s in series_ids}
    for i, (chunk, score) in enumerate(zip(chunks, raw_scores)):
        s = float(score)
        if chunk.source == "concepts.md":
            s *= _CONCEPTS_BOOST
        # Hard bonus if the chunk explicitly contains a series ID token.
        if any(sid in chunk.text.upper() for sid in series_upper):
            s += 5.0
        boosted.append((s, i))

    boosted.sort(key=lambda x: x[0], reverse=True)
    selected = [chunks[i] for _, i in boosted[:top_n]]
    return _truncate_chunks(selected, max_chars_per_chunk)


def _truncate_chunks(chunks: list[Chunk], max_chars: int) -> list[Chunk]:
    """Return copies of chunks with text truncated to max_chars."""
    if max_chars <= 0:
        return chunks
    out: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= max_chars:
            out.append(c)
        else:
            truncated = c.text[:max_chars].rstrip() + "\n… [truncated]"
            out.append(Chunk(source=c.source, header=c.header, text=truncated))
    return out
