"""Extract text from PDF files using PyMuPDF, preserving page structure."""

from __future__ import annotations

import re

import fitz  # PyMuPDF

# Patterns for repeated headers/footers to strip
_STRIP_PATTERNS = [
    # EC2 copyright footer
    re.compile(
        r"Copyright Ente Nazionale Italiano di Unificazione\s*\n"
        r"Provided by IHS under license with UNI\s*\n"
        r".*?Licensee=.*?\n"
        r".*?Not for Resale.*?\n"
        r".*?No reproduction.*",
        re.DOTALL,
    ),
    # EC2 page footer: "UNI EN 1992-1-1:2005    © UNI    Pagina 17"
    re.compile(r"UNI EN \d{4}-\d+-\d+:\d{4}\s+© UNI\s+Pagina \d+"),
    # NTC page header: "20-2-2018    Supplemento ordinario n. 8 alla GAZZETTA UFFICIALE    Serie generale - n. 42"
    re.compile(r"\d+-\d+-\d{4}\s+Supplemento ordinario.*?Serie generale.*?n\.\s*\d+"),
    # NTC page number footer: "— 6 —"
    re.compile(r"—\s*\d+\s*—"),
]


def _clean_page_text(text: str) -> str:
    """Remove known headers/footers from page text."""
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF.

    Returns a list of dicts with keys: page_number, text.
    Page numbers are 1-based (as printed in the document).
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = _clean_page_text(text)
        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text,
            })
    doc.close()
    return pages


def extract_full_text(pdf_path: str) -> str:
    """Extract all text from a PDF as a single string with page markers."""
    pages = extract_pages(pdf_path)
    parts = []
    for p in pages:
        parts.append(f"\n\n[PAGE {p['page_number']}]\n\n{p['text']}")
    return "\n".join(parts)
