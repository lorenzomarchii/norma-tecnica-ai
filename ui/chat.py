"""Chat interface components."""

from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

logger = logging.getLogger("norma-tecnica-ai")
logging.basicConfig(level=logging.INFO)

from generation.llm_client import generate_answer_stream
from retrieval.hybrid_retriever import retrieve
from ui.citations import render_citations


def render_chat(document_filter: str | None = None) -> None:
    """Render the chat interface and handle user input."""
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "chunks" in msg:
                render_citations(msg["content"], msg["chunks"])

    # Chat input
    if prompt := st.chat_input("Fai una domanda sulla normativa..."):
        logger.info(f"[QUERY] {datetime.now().isoformat()} | {prompt}")
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Retrieve relevant chunks
            with st.spinner("Cerco nei documenti..."):
                chunks = retrieve(prompt, document_filter=document_filter)

            if not chunks:
                answer = "Non ho trovato documenti indicizzati. Carica prima i PDF dalla sidebar."
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": [],
                })
                return

            # Build conversation history for multi-turn (last 6 messages)
            history = []
            for msg in st.session_state.messages[-6:]:
                if msg["role"] in ("user", "assistant"):
                    history.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })
            # Remove the last user message (it's included in the prompt)
            if history and history[-1]["role"] == "user":
                history.pop()

            # Stream the response
            answer = st.write_stream(
                generate_answer_stream(
                    query=prompt,
                    chunks=chunks,
                    conversation_history=history if history else None,
                )
            )

            render_citations(answer, chunks)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": chunks,
            })
