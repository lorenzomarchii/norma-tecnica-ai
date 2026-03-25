"""Hybrid retrieval combining vector search and BM25 with Reciprocal Rank Fusion."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from config import settings
from ingestion.embedder import get_embeddings, get_single_embedding
from retrieval import bm25_store, vector_store

logger = logging.getLogger("norma-tecnica-ai")


def _expand_query(query: str) -> list[str]:
    """Use Claude to expand a broad query into specific sub-queries for better retrieval."""
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0,
            system="Sei un ingegnere strutturista. Data una domanda, genera 3 sotto-query specifiche per cercare nelle NTC 2018 e nell'Eurocodice 2. Rispondi SOLO con le query, una per riga, senza numerazione o prefissi.",
            messages=[{"role": "user", "content": query}],
        )
        sub_queries = [q.strip() for q in response.content[0].text.strip().split("\n") if q.strip()][:3]
        logger.info(f"[MULTI-QUERY] Expanded into {len(sub_queries)} sub-queries")
        return sub_queries
    except Exception as e:
        logger.warning(f"[MULTI-QUERY] Failed: {e}")
        return []


def _search_single_query(query_embedding, query_text, document_filter):
    """Search vector + BM25 for a single query."""
    vr = vector_store.search(
        query_embedding=query_embedding,
        top_k=settings.vector_top_k,
        document_filter=document_filter,
    )
    br = bm25_store.search(
        query=query_text,
        top_k=settings.bm25_top_k,
        document_filter=document_filter,
    )
    return vr, br


def retrieve(
    query: str,
    document_filter: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Hybrid retrieval: vector + BM25 with RRF fusion + cross-ref expansion.

    Uses multi-query expansion and parallel search for speed.
    """
    top_k = top_k or settings.final_top_k

    # Expand query into sub-queries for better coverage
    sub_queries = _expand_query(query)
    all_queries = [query] + sub_queries

    # Batch all embeddings in one API call (much faster than sequential)
    all_embeddings = get_embeddings(all_queries)

    # Search in parallel for all queries
    all_vector_results = []
    all_bm25_results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, q in enumerate(all_queries):
            futures.append(executor.submit(
                _search_single_query, all_embeddings[i], q, document_filter
            ))
        for future in as_completed(futures):
            vr, br = future.result()
            all_vector_results.extend(vr)
            all_bm25_results.extend(br)

    # Reciprocal Rank Fusion across all results
    fused = _reciprocal_rank_fusion(all_vector_results, all_bm25_results, k=60)

    # Take more results since we have broader coverage now
    expanded_top_k = min(top_k + 4, len(fused))
    top_results = fused[:expanded_top_k]

    # Cross-reference expansion
    expanded = _expand_cross_references(top_results, document_filter)

    # Order by document name then section number for logical reading order
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
