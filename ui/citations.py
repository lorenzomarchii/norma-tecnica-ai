"""Citation rendering components for the chat UI."""

from __future__ import annotations

import streamlit as st

from generation.citation_validator import validate_citations


def render_citations(answer: str, chunks: list[dict]) -> None:
    """Render an expandable citation section below the answer."""
    citations = validate_citations(answer, chunks)

    if not citations and not chunks:
        return

    with st.expander("📋 Fonti consultate", expanded=False):
        if citations:
            for c in citations:
                icon = "✅" if c["valid"] else "⚠️"
                st.markdown(f"{icon} {c['reference']}")
        else:
            # Show all chunks used as context even if no explicit citations were parsed
            for chunk in chunks:
                pages = ", ".join(str(p) for p in chunk["page_numbers"])
                st.markdown(
                    f"📄 **{chunk['document_name']}**, "
                    f"par. {chunk['section_number']} — *{chunk['section_title']}* "
                    f"(pag. {pages})"
                )
