"""Validate citations in generated answers against provided chunks."""

from __future__ import annotations

import re


def validate_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Check that citations in the answer match provided chunks.

    Returns a list of citation dicts with keys: reference, valid, section_number.
    """
    # Extract all citations like [NTC 2018, par. 4.1.2, pag. 45] or [EC2, 6.2.3, pag. 89]
    citation_pattern = re.compile(
        r"\[([^]]+?,\s*(?:par\.|§)\s*([\d.]+[^,]*),\s*pag\.\s*[\d, ]+)\]"
    )

    available_sections = {c["section_number"].split("_")[0] for c in chunks}

    citations = []
    for match in citation_pattern.finditer(answer):
        full_ref = match.group(1)
        section_num = match.group(2).strip().rstrip(".")

        # Check if the cited section exists in the provided chunks
        valid = any(
            section_num == s or s.startswith(section_num + ".") or section_num.startswith(s + ".")
            for s in available_sections
        )

        citations.append({
            "reference": full_ref,
            "section_number": section_num,
            "valid": valid,
        })

    return citations
