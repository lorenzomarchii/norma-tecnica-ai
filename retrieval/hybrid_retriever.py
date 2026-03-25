"""Hybrid retrieval combining vector search and BM25 with Reciprocal Rank Fusion."""

from __future__ import annotations

import json

from config import settings
from ingestion.embedder import get_single_embedding
from retrieval import bm25_store, vector_store


def retrieve(
    query: str,
    document_filter: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Hybrid retrieval: vector + BM25 with RRF fusion + cross-ref expansion.

    Args:
        query: User query string.
        document_filter: Optional filter by document name.
        top_k: Number of final results.

    Returns:
        List of result dicts ordered by document then section, with keys:
        section_number, section_title, document_name, text, page_numbers, score.
    """
    top_k = top_k or settings.final_top_k

    # 1. Vector search
    query_embedding = get_single_embedding(query)
    vector_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=settings.vector_top_k,
        document_filter=document_filter,
    )

    # 2. BM25 search
    bm25_results = bm25_store.search(
        query=query,
        top_k=settings.bm25_top_k,
        document_filter=document_filter,
    )

    # 3. Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(vector_results, bm25_results, k=60)

    # 4. Take top results
    top_results = fused[:top_k]

    # 5. Cross-reference expansion
    expanded = _expand_cross_references(top_results, document_filter)

    # 6. Order by document name then section number for logical reading order
    expanded.sort(key=lambda r: (r["document_name"], _section_sort_key(r["section_number"])))

    return expanded


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists using RRF: score(d) = sum(1 / (k + rank))."""
    # Build a unified map keyed by a unique identifier
    doc_map: dict[str, dict] = {}

    # Process vector results
    for rank, vr in enumerate(vector_results):
        key = vr["id"]
        if key not in doc_map:
            page_numbers = json.loads(vr["metadata"].get("page_numbers", "[]"))
            doc_map[key] = {
                "section_number": vr["metadata"]["section_number"],
                "section_title": vr["metadata"]["section_title"],
                "document_name": vr["metadata"]["document_name"],
                "text": vr["text"],
                "page_numbers": page_numbers,
                "cross_references": json.loads(vr["metadata"].get("cross_references", "[]")),
                "rrf_score": 0.0,
            }
        doc_map[key]["rrf_score"] += 1.0 / (k + rank)

    # Process BM25 results
    for rank, br in enumerate(bm25_results):
        chunk = br["chunk"]
        key = chunk.chunk_id
        if key not in doc_map:
            doc_map[key] = {
                "section_number": chunk.section_number,
                "section_title": chunk.section_title,
                "document_name": chunk.document_name,
                "text": chunk.text,
                "page_numbers": chunk.page_numbers,
                "cross_references": chunk.cross_references,
                "rrf_score": 0.0,
            }
        doc_map[key]["rrf_score"] += 1.0 / (k + rank)

    # Sort by RRF score descending
    results = sorted(doc_map.values(), key=lambda x: x["rrf_score"], reverse=True)
    return results


def _expand_cross_references(
    results: list[dict],
    document_filter: str | None = None,
) -> list[dict]:
    """Add chunks referenced by the top results via cross-references."""
    max_expansion = settings.max_cross_ref_expansion
    existing_sections = {r["section_number"] for r in results}
    expanded = list(results)

    # Collect cross-references from top 3 results
    refs_to_fetch: set[str] = set()
    for r in results[:3]:
        for ref in r.get("cross_references", []):
            if ref not in existing_sections:
                refs_to_fetch.add(ref)

    if not refs_to_fetch:
        return expanded

    # Search for referenced sections via BM25 (exact section number match)
    added = 0
    for ref in refs_to_fetch:
        if added >= max_expansion:
            break
        ref_results = bm25_store.search(
            query=ref,
            top_k=3,
            document_filter=document_filter,
        )
        for rr in ref_results:
            chunk = rr["chunk"]
            if chunk.section_number.startswith(ref) and chunk.section_number not in existing_sections:
                expanded.append({
                    "section_number": chunk.section_number,
                    "section_title": chunk.section_title,
                    "document_name": chunk.document_name,
                    "text": chunk.text,
                    "page_numbers": chunk.page_numbers,
                    "cross_references": chunk.cross_references,
                    "rrf_score": 0.0,
                })
                existing_sections.add(chunk.section_number)
                added += 1
                break

    return expanded


def _section_sort_key(section_number: str) -> list:
    """Convert section number to a sortable key.

    "4.1.2" -> [4, 1, 2], handles "_p1" suffixes.
    """
    # Remove part suffix like "_p1"
    base = section_number.split("_")[0]
    parts = []
    for p in base.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return parts
