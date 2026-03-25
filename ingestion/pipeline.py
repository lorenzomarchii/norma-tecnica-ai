"""Orchestrate the full ingestion pipeline: parse -> chunk -> embed -> store."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from config import DATA_DIR, settings
from ingestion.chunker import Chunk, chunk_document
from ingestion.embedder import get_embeddings
from ingestion.pdf_parser import extract_pages


def ingest_pdf(
    pdf_path: str,
    document_name: str | None = None,
    collection_name: str = "normative_chunks",
) -> list[Chunk]:
    """Full ingestion pipeline for a single PDF.

    1. Extract pages
    2. Chunk by section
    3. Generate embeddings
    4. Store in ChromaDB + BM25 index

    Returns the list of created chunks.
    """
    print(f"[1/4] Extracting pages from {pdf_path}...")
    pages = extract_pages(pdf_path)
    print(f"       Extracted {len(pages)} pages.")

    print("[2/4] Chunking document by sections...")
    chunks = chunk_document(pages, document_name=document_name)
    print(f"       Created {len(chunks)} chunks.")

    print("[3/4] Generating embeddings...")
    texts = [f"{c.section_number} {c.section_title}\n\n{c.text}" for c in chunks]
    embeddings = get_embeddings(texts)
    print(f"       Generated {len(embeddings)} embeddings.")

    print("[4/4] Storing in ChromaDB + BM25 index...")
    _store_in_chroma(chunks, embeddings, collection_name)
    _update_bm25_index(chunks)
    _save_chunks_json(chunks)
    print("       Done!")

    return chunks


def _store_in_chroma(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    collection_name: str,
) -> None:
    """Store chunks and embeddings in ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = chunk.chunk_id
        # Handle duplicate IDs by appending a counter
        base_id = chunk_id
        counter = 1
        while chunk_id in ids:
            chunk_id = f"{base_id}_{counter}"
            counter += 1

        ids.append(chunk_id)
        documents.append(chunk.text)
        metadatas.append({
            "document_name": chunk.document_name,
            "section_number": chunk.section_number,
            "section_title": chunk.section_title,
            "page_numbers": json.dumps(chunk.page_numbers),
            "parent_sections": json.dumps(chunk.parent_sections),
            "cross_references": json.dumps(chunk.cross_references),
            "chunk_type": chunk.chunk_type,
        })

    # Upsert in batches (ChromaDB limit)
    batch_size = 166
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )


def _update_bm25_index(chunks: list[Chunk]) -> None:
    """Update or create the BM25 index."""
    from rank_bm25 import BM25Okapi

    bm25_path = DATA_DIR / "bm25_index.pkl"
    chunks_for_bm25_path = DATA_DIR / "bm25_chunks.pkl"

    # Load existing chunks if any
    existing_chunks: list[Chunk] = []
    if chunks_for_bm25_path.exists():
        with open(chunks_for_bm25_path, "rb") as f:
            existing_chunks = pickle.load(f)

    # Merge: remove chunks from same document, add new ones
    doc_name = chunks[0].document_name if chunks else None
    existing_chunks = [c for c in existing_chunks if c.document_name != doc_name]
    all_chunks = existing_chunks + chunks

    # Build BM25 index on tokenized text
    tokenized = [_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(chunks_for_bm25_path, "wb") as f:
        pickle.dump(all_chunks, f)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing."""
    import re
    return re.findall(r"\w+", text.lower())


def _save_chunks_json(chunks: list[Chunk]) -> None:
    """Save chunks as JSON for inspection/debugging."""
    out_path = DATA_DIR / "chunks"
    out_path.mkdir(exist_ok=True)

    doc_name = chunks[0].document_name if chunks else "unknown"
    safe_name = doc_name.replace(" ", "_").replace("/", "_")
    file_path = out_path / f"{safe_name}.json"

    data = []
    for c in chunks:
        data.append({
            "chunk_id": c.chunk_id,
            "document_name": c.document_name,
            "section_number": c.section_number,
            "section_title": c.section_title,
            "text": c.text[:500] + "..." if len(c.text) > 500 else c.text,
            "text_length": len(c.text),
            "page_numbers": c.page_numbers,
            "parent_sections": c.parent_sections,
            "cross_references": c.cross_references,
            "chunk_type": c.chunk_type,
        })

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"       Chunks saved to {file_path}")
