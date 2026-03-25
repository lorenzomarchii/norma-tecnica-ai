# NormaTecnica AI

## Cosa è
MVP di un sistema RAG per interrogare norme tecniche italiane per l'ingegneria civile (NTC 2018 + Eurocodice 2). L'utente carica i PDF delle norme, il sistema li indicizza e permette di fare domande con citazioni precise delle clausole.

## Obiettivo business
Testare il prodotto gratuitamente con uno studio di ingegneria civile, poi scalare come SaaS a pagamento ad altri studi.

## Tech Stack
- **LLM**: Claude Sonnet (claude-sonnet-4-6) via Anthropic API
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB (embedded, persistente)
- **Keyword Search**: rank_bm25 (ricerca ibrida)
- **Frontend**: Streamlit
- **Deploy previsto**: Railway (~$5/mese)
- **PDF Parsing**: PyMuPDF
- **Python**: 3.9+ (usare `from __future__ import annotations` in tutti i file)

## Struttura progetto
```
app.py                    → Entry point Streamlit
config.py                 → Settings (pydantic-settings, legge da .env)
ingestion/
  pdf_parser.py           → Estrazione testo da PDF + pulizia header/footer
  chunker.py              → Chunking per clausola/sezione con metadata
  cross_ref_parser.py     → Parsing riferimenti incrociati tra sezioni
  embedder.py             → Chiamate embedding OpenAI
  pipeline.py             → Orchestratore: parse → chunk → embed → store
retrieval/
  vector_store.py         → Wrapper ChromaDB
  bm25_store.py           → Indice BM25 per keyword search
  hybrid_retriever.py     → Fusione RRF (vector + BM25) + cross-ref expansion
generation/
  prompts.py              → System prompt e template per Claude
  llm_client.py           → Client Anthropic API con streaming
  citation_validator.py   → Validazione citazioni nella risposta
ui/
  sidebar.py              → Upload PDF, filtri documento, nuova chat
  chat.py                 → Interfaccia chat con streaming
  citations.py            → Rendering citazioni espandibili
```

## Documenti supportati
- **NTC 2018** (Norme Tecniche per le Costruzioni) — pubbliche, da Gazzetta Ufficiale, ~372 pagine
- **Eurocodice 2** (UNI EN 1992-1-1:2005) — copyrighted UNI, ~224 pagine, modello BYOL (l'utente carica il proprio PDF)

## Come funziona la pipeline
1. PDF → PyMuPDF estrae testo pagina per pagina, rimuove header/footer ripetuti
2. Chunker detecta tipo documento (NTC vs EC), identifica sezioni per numerazione gerarchica, splitta per clausola
3. Ogni chunk ha metadata: document_name, section_number, section_title, page_numbers, parent_sections, cross_references
4. Embedding con OpenAI → salvati in ChromaDB + indice BM25
5. Query: ricerca ibrida (vector + BM25) → fusione Reciprocal Rank Fusion → espansione cross-reference → Claude Sonnet genera risposta con citazioni

## Comandi utili
```bash
# Lanciare l'app
cd /Users/Lorenzo/norma-tecnica-ai && streamlit run app.py

# Ingestire manualmente un PDF (da Python)
from ingestion.pipeline import ingest_pdf
ingest_pdf('/path/to/pdf.pdf')
```

## Note importanti
- Il file `.env` contiene le API key (ANTHROPIC_API_KEY + OPENAI_API_KEY) — mai committare
- Python 3.9: SEMPRE aggiungere `from __future__ import annotations` in testa ai file .py
- Le pagine TOC e frontmatter vengono filtrate automaticamente dal chunker
- Il system prompt impone a Claude di rispondere SOLO dal contesto e citare SEMPRE clausola/pagina
- Temperature 0 per risposte deterministiche
