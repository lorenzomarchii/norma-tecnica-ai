"""Generate embeddings using OpenRouter API (OpenAI-compatible)."""

from __future__ import annotations

import requests

from config import settings


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts via OpenRouter."""
    all_embeddings = []
    batch_size = 256

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.embedding_model,
                "input": batch,
            },
        )
        response.raise_for_status()
        data = response.json()
        batch_embeddings = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embeddings)

        if len(texts) > batch_size:
            done = min(i + batch_size, len(texts))
            print(f"       Embeddings: {done}/{len(texts)}")

    return all_embeddings


def get_single_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return get_embeddings([text])[0]
