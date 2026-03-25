"""Sidebar components: PDF upload, document filters, chat history."""

from __future__ import annotations

import streamlit as st


def render_sidebar() -> dict:
    """Render the sidebar and return the current settings.

    Returns dict with keys: document_filter, uploaded_files.
    """
    with st.sidebar:
        st.title("NormaTecnica AI")
        st.caption("Assistente per normativa tecnica italiana")

        st.divider()

        # PDF Upload
        st.subheader("Carica documenti")
        uploaded_files = st.file_uploader(
            "Carica PDF delle norme",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        # Show indexed documents
        if "indexed_docs" in st.session_state and st.session_state.indexed_docs:
            st.subheader("Documenti indicizzati")
            for doc_name, chunk_count in st.session_state.indexed_docs.items():
                st.success(f"**{doc_name}** — {chunk_count} sezioni", icon="✅")

        st.divider()

        # Document filter
        st.subheader("Filtra per documento")
        filter_options = ["Tutti i documenti"]
        if "indexed_docs" in st.session_state:
            filter_options.extend(st.session_state.indexed_docs.keys())

        selected_filter = st.radio(
            "Cerca in:",
            filter_options,
            index=0,
            label_visibility="collapsed",
        )

        document_filter = None if selected_filter == "Tutti i documenti" else selected_filter

        st.divider()

        # New chat button
        if st.button("Nuova conversazione", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return {
        "document_filter": document_filter,
        "uploaded_files": uploaded_files,
    }
