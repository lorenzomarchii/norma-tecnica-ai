"""BM25 keyword search index."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import DATA_DIR
from ingestion.chunker import Chunk

BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
BM25_CHUNKS_PATH = DATA_DIR / "bm25_chunks.pkl"


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return re.findall(r"\w+", text.lower())


def load_index() -> tuple[BM25Okapi | None, list[Chunk]]:
    """Load the BM25 index and associated chunks from disk."""
    if not BM25_INDEX_PATH.exists() or not BM25_CHUNKS_PATH.exists():
        return None, []

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return bm25, chunks


def search(
    query: str,
    top_k: int | None = None,
    document_filter: str | None = None,
) -> list[dict]:
    """Search using BM25 keyword matching.

    Args:
        query: The search query string.
        top_k: Number of results to return.
        document_filter: Optional filter by document_name.

    Returns:
        List of result dicts with keys: chunk, score, rank.
    """
    from config import settings

    top_k = top_k or settings.bm25_top_k
    bm25, chunks = load_index()

    if bm25 is None or not chunks:
        return []

    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    # Pair scores with chunks and sort
    scored = list(zip(scores, chunks))
    if document_filter:
        scored = [(s, c) for s, c in scored if c.document_name == document_filter]

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for rank, (score, chunk) in enumerate(scored[:top_k]):
        if score <= 0:
            break
        results.append({
            "chunk": chunk,
            "score": float(score),
            "rank": rank,
        })

    return results
