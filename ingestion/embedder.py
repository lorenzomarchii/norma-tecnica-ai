"""Generate embeddings using a local sentence-transformers model."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from config import settings

# Lazy-loaded model singleton
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Uses a local multilingual model — no API calls, no cost.
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings.tolist()


def get_single_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    model = _get_model()
    embedding = model.encode(text)
    return embedding.tolist()
