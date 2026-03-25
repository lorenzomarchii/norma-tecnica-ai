"""NormaTecnica AI — Main Streamlit application."""

import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="NormaTecnica AI",
    page_icon="🏗️",
    layout="wide",
)

from config import settings
from ui.chat import render_chat
from ui.sidebar import render_sidebar


def _process_uploaded_file(uploaded_file) -> None:
    """Save uploaded PDF to temp dir, run ingestion pipeline, update state."""
    from ingestion.pipeline import ingest_pdf

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # Run ingestion
    with st.spinner(f"Elaborazione di {uploaded_file.name}..."):
        chunks = ingest_pdf(tmp_path)

    if chunks:
        doc_name = chunks[0].document_name
        if "indexed_docs" not in st.session_state:
            st.session_state.indexed_docs = {}
        st.session_state.indexed_docs[doc_name] = len(chunks)

        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        st.session_state.processed_files.add(uploaded_file.name)

    # Cleanup temp file
    Path(tmp_path).unlink(missing_ok=True)


def main():
    # Check API key
    if not settings.anthropic_api_key:
        st.error(
            "⚠️ Configura la API key nel file `.env`:\n"
            "- `ANTHROPIC_API_KEY` per Claude Sonnet"
        )
        st.stop()

    # Detect already-indexed documents from ChromaDB
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = {}
        try:
            from retrieval.vector_store import get_collection
            collection = get_collection()
            if collection.count() > 0:
                # Get unique document names from metadata
                results = collection.get(limit=1, include=["metadatas"])
                # Peek at all docs to count per-document
                all_meta = collection.get(include=["metadatas"])
                doc_counts = {}
                for meta in all_meta["metadatas"]:
                    name = meta.get("document_name", "Sconosciuto")
                    doc_counts[name] = doc_counts.get(name, 0) + 1
                st.session_state.indexed_docs = doc_counts
        except Exception:
            pass

    # Render sidebar and get settings
    sidebar_state = render_sidebar()

    # Process uploaded files
    if sidebar_state["uploaded_files"]:
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()

        for uploaded_file in sidebar_state["uploaded_files"]:
            if uploaded_file.name not in st.session_state.processed_files:
                _process_uploaded_file(uploaded_file)
                st.rerun()

    # Main chat area
    render_chat(document_filter=sidebar_state["document_filter"])


if __name__ == "__main__":
    main()
