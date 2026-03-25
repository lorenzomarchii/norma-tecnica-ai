"""ChromaDB vector store wrapper."""

from __future__ import annotations

import json

import chromadb

from config import settings


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection(
    name: str = "normative_chunks",
    client: chromadb.PersistentClient | None = None,
) -> chromadb.Collection:
    client = client or get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def search(
    query_embedding: list[float],
    top_k: int | None = None,
    document_filter: str | None = None,
    collection_name: str = "normative_chunks",
) -> list[dict]:
    """Search the vector store by embedding similarity.

    Args:
        query_embedding: The query vector.
        top_k: Number of results to return.
        document_filter: Optional filter by document_name.

    Returns:
        List of result dicts with keys: id, text, metadata, distance.
    """
    top_k = top_k or settings.vector_top_k
    collection = get_collection(collection_name)

    where_filter = None
    if document_filter:
        where_filter = {"document_name": document_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    items = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            items.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
    return items
