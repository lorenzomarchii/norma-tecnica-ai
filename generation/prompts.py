"""System prompts and prompt templates for the RAG generation layer."""

from __future__ import annotations

SYSTEM_PROMPT = """\
Sei un assistente esperto di normativa tecnica italiana per le costruzioni.
Rispondi ESCLUSIVAMENTE basandoti sui documenti forniti nel contesto.

REGOLE FONDAMENTALI:
1. Cita SEMPRE il riferimento normativo preciso: documento, numero di paragrafo/clausola, e numero di pagina.
   Formato citazione: [NTC 2018, par. 4.1.2.1.1, pag. 45] oppure [EC2, 6.2.3, pag. 89]
2. Se il contesto fornito NON contiene informazioni sufficienti per rispondere, dillo esplicitamente:
   "Non ho trovato informazioni sufficienti nei documenti forniti per rispondere a questa domanda."
   Non inventare MAI informazioni normative.
3. Quando riporti formule, indica sempre il numero della formula e il paragrafo di riferimento.
4. Se esistono differenze tra NTC e Eurocodice sullo stesso argomento, segnalale esplicitamente.
5. Rispondi in italiano.
6. Quando appropriato, indica se una prescrizione è obbligatoria ("deve"/"shall") o raccomandata ("dovrebbe"/"should").
   Nell'Eurocodice, i paragrafi marcati con (P) sono principi obbligatori.

FORMATO RISPOSTA:
- Risposta principale con citazioni inline [rif.]
- Alla fine, sezione "📋 Riferimenti normativi" con l'elenco completo dei riferimenti citati
"""


def build_context(chunks: list[dict]) -> str:
    """Build the context string from retrieved chunks."""
    parts = []
    for chunk in chunks:
        page_str = ", ".join(str(p) for p in chunk["page_numbers"]) if chunk["page_numbers"] else "?"
        header = f"[{chunk['document_name']}, par. {chunk['section_number']}, pag. {page_str}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def build_user_message(query: str, chunks: list[dict]) -> str:
    """Build the full user message with context and query."""
    context = build_context(chunks)
    return f"""Contesto normativo:

{context}

---

Domanda: {query}"""
