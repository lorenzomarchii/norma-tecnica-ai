"""Section-based chunking for Italian technical standards (NTC and Eurocode)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingestion.cross_ref_parser import extract_cross_references

# ---------------------------------------------------------------------------
# Section header patterns
# ---------------------------------------------------------------------------

# NTC 2018: "CAPITOLO 2 – SICUREZZA ..." or "2.1. PRINCIPI FONDAMENTALI"
_NTC_CHAPTER = re.compile(
    r"^CAPITOLO\s+(\d+)\s*[–\-]\s*(.+)$", re.MULTILINE
)
_NTC_SECTION = re.compile(
    r"^(\d+(?:\.\d+)+)\.?\s+([A-Z\u00C0-\u00FF].{2,})$", re.MULTILINE
)

# EC2: "SEZIONE 3    MATERIALI" or "3.1    Calcestruzzo" or "3.1.1    Generalità"
_EC2_SECTION_HEADER = re.compile(
    r"^SEZIONE\s+(\d+)\s+(.+)$", re.MULTILINE
)
# EC2 clause: must NOT be followed by dots (TOC lines) and section numbers must start 1-12
_EC2_CLAUSE = re.compile(
    r"^(\d{1,2}(?:\.\d+)+)\s{2,}([A-Z\u00C0-\u00FF].{2,})$", re.MULTILINE
)

# Pattern to detect TOC lines: "3.1.2  Resistenza ........... 16"
_TOC_LINE = re.compile(r"\.{4,}\s*\d+\s*$")


@dataclass
class Chunk:
    """A chunk of text from a technical standard with metadata."""

    document_name: str
    section_number: str
    section_title: str
    text: str
    page_numbers: list[int] = field(default_factory=list)
    parent_sections: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    chunk_type: str = "text"  # text, table, note

    @property
    def chunk_id(self) -> str:
        return f"{self.document_name}::{self.section_number}"


def _compute_parents(section_number: str) -> list[str]:
    """Compute parent section numbers.

    Example: "4.1.2.1" -> ["4", "4.1", "4.1.2"]
    """
    parts = section_number.split(".")
    parents = []
    for i in range(1, len(parts)):
        parents.append(".".join(parts[:i]))
    return parents


def _is_toc_or_frontmatter(page_text: str) -> bool:
    """Detect if a page is a table of contents or frontmatter page."""
    # TOC pages have many dot-leader lines (dots may be on separate lines from page numbers)
    dot_lines = len(re.findall(r"\.{5,}", page_text))
    if dot_lines >= 3:
        return True
    # Frontmatter: pages with Roman numeral page numbers (Pagina I, II, ..., XII, etc.)
    if re.search(r"Pagina\s+[IVXLC]+\s*$", page_text, re.MULTILINE):
        return True
    # Other frontmatter markers (need at least 2 to avoid false positives)
    frontmatter_markers = [
        "Riproduzione vietata", "PREMESSA NAZIONALE",
        "SOMMARIO", "DECRETI, DELIBERE E ORDINANZE",
    ]
    marker_count = sum(1 for m in frontmatter_markers if m in page_text)
    if marker_count >= 1 and len(page_text.strip()) < 2000:
        return True
    return False


def _detect_document_type(full_text: str) -> str:
    """Heuristic to detect if a PDF is NTC or Eurocode."""
    if "Gazzetta Ufficiale" in full_text[:3000] or "CAPITOLO" in full_text[:5000]:
        return "NTC"
    if "UNI EN" in full_text[:3000] or "Eurocodice" in full_text[:3000]:
        return "EC"
    return "UNKNOWN"


def _extract_document_name(full_text: str, doc_type: str) -> str:
    if doc_type == "NTC":
        return "NTC 2018"
    if doc_type == "EC":
        # Try to extract the specific EN number
        m = re.search(r"UNI EN (\d{4}-\d+-\d+)", full_text[:2000])
        if m:
            return f"Eurocodice 2 (UNI EN {m.group(1)})"
        return "Eurocodice 2"
    return "Documento sconosciuto"


def _find_page_for_position(page_markers: list[tuple[int, int]], pos: int) -> int:
    """Given a position in the full text, return the page number."""
    current_page = 1
    for marker_pos, page_num in page_markers:
        if marker_pos <= pos:
            current_page = page_num
        else:
            break
    return current_page


def _parse_page_markers(full_text: str) -> list[tuple[int, int]]:
    """Find all [PAGE N] markers and their positions."""
    markers = []
    for m in re.finditer(r"\[PAGE (\d+)\]", full_text):
        markers.append((m.start(), int(m.group(1))))
    return markers


def chunk_document(pages: list[dict], document_name: str | None = None) -> list[Chunk]:
    """Chunk a document into sections based on its structure.

    Args:
        pages: List of page dicts from pdf_parser.extract_pages().
        document_name: Override for the document name. If None, auto-detected.

    Returns:
        List of Chunk objects.
    """
    # Filter out TOC and frontmatter pages
    content_pages = []
    for p in pages:
        if _is_toc_or_frontmatter(p["text"]):
            continue
        content_pages.append(p)

    # Build full text with page markers
    parts = []
    for p in content_pages:
        parts.append(f"\n[PAGE {p['page_number']}]\n{p['text']}")
    full_text = "\n".join(parts)

    page_markers = _parse_page_markers(full_text)
    doc_type = _detect_document_type(full_text)
    if document_name is None:
        document_name = _extract_document_name(full_text, doc_type)

    # Find all section headers with their positions
    headers: list[tuple[int, str, str]] = []  # (position, section_number, title)

    if doc_type == "NTC":
        for m in _NTC_CHAPTER.finditer(full_text):
            headers.append((m.start(), m.group(1), m.group(2).strip()))
        for m in _NTC_SECTION.finditer(full_text):
            sec_num = m.group(1).rstrip(".")
            title = m.group(2).strip()
            # Skip TOC entries (title followed by dots and page number)
            if _TOC_LINE.search(title):
                continue
            headers.append((m.start(), sec_num, title))
    else:  # EC or UNKNOWN
        for m in _EC2_SECTION_HEADER.finditer(full_text):
            headers.append((m.start(), m.group(1), m.group(2).strip()))
        for m in _EC2_CLAUSE.finditer(full_text):
            title = m.group(2).strip()
            if _TOC_LINE.search(title):
                continue
            headers.append((m.start(), m.group(1), title))

    if not headers:
        # Fallback: treat entire document as one chunk
        clean_text = re.sub(r"\[PAGE \d+\]", "", full_text).strip()
        return [
            Chunk(
                document_name=document_name,
                section_number="1",
                section_title="Documento completo",
                text=clean_text,
                page_numbers=list(range(1, len(pages) + 1)),
            )
        ]

    # Sort headers by position in text
    headers.sort(key=lambda h: h[0])

    # Deduplicate headers at same position (keep the one with more specific section number)
    deduped: list[tuple[int, str, str]] = []
    for h in headers:
        if deduped and abs(h[0] - deduped[-1][0]) < 5:
            # Keep the more specific one (longer section number)
            if len(h[1]) > len(deduped[-1][1]):
                deduped[-1] = h
        else:
            deduped.append(h)
    headers = deduped

    # Create chunks between consecutive headers
    chunks: list[Chunk] = []
    for i, (pos, sec_num, title) in enumerate(headers):
        # Text runs from this header to the next header (or end of document)
        end_pos = headers[i + 1][0] if i + 1 < len(headers) else len(full_text)
        raw_text = full_text[pos:end_pos]

        # Clean page markers from text
        clean_text = re.sub(r"\[PAGE \d+\]", "", raw_text).strip()

        if not clean_text or len(clean_text) < 10:
            continue

        # Determine page range
        start_page = _find_page_for_position(page_markers, pos)
        end_page = _find_page_for_position(page_markers, end_pos - 1)
        page_range = list(range(start_page, end_page + 1))

        chunk = Chunk(
            document_name=document_name,
            section_number=sec_num,
            section_title=title,
            text=clean_text,
            page_numbers=page_range,
            parent_sections=_compute_parents(sec_num),
            cross_references=extract_cross_references(clean_text),
        )
        chunks.append(chunk)

    # Split chunks that are too large (>2000 chars) into sub-chunks
    final_chunks: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) <= 2000:
            final_chunks.append(chunk)
        else:
            sub_chunks = _split_large_chunk(chunk)
            final_chunks.extend(sub_chunks)

    return final_chunks


def _split_large_chunk(chunk: Chunk, max_size: int = 2000) -> list[Chunk]:
    """Split a large chunk into smaller pieces at paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", chunk.text)

    sub_chunks = []
    current_text = ""
    part_num = 0

    for para in paragraphs:
        if len(current_text) + len(para) > max_size and current_text:
            part_num += 1
            sub_chunks.append(
                Chunk(
                    document_name=chunk.document_name,
                    section_number=f"{chunk.section_number}_p{part_num}",
                    section_title=chunk.section_title,
                    text=current_text.strip(),
                    page_numbers=chunk.page_numbers,
                    parent_sections=chunk.parent_sections,
                    cross_references=extract_cross_references(current_text),
                )
            )
            current_text = para
        else:
            current_text += "\n\n" + para if current_text else para

    if current_text.strip():
        part_num += 1
        suffix = f"_p{part_num}" if part_num > 1 or sub_chunks else ""
        sub_chunks.append(
            Chunk(
                document_name=chunk.document_name,
                section_number=f"{chunk.section_number}{suffix}",
                section_title=chunk.section_title,
                text=current_text.strip(),
                page_numbers=chunk.page_numbers,
                parent_sections=chunk.parent_sections,
                cross_references=extract_cross_references(current_text),
            )
        )

    return sub_chunks
