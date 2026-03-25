"""Extract cross-references to other sections/clauses from chunk text."""

from __future__ import annotations

import re

# Patterns for cross-references in Italian technical standards
_PATTERNS = [
    # "par. 4.1.2.1" or "paragrafo 4.1.2"
    r"(?:par(?:\.|agrafo)?|punto|sezione|capitolo|clausola)\s+(\d+(?:\.\d+)+(?:\s*\(\d+\))?)",
    # "Sezione 6" or "Capitolo 4"
    r"(?:Sezione|Capitolo)\s+(\d+)",
    # "prospetto 3.1" or "figura 6.2"
    r"(?:prospetto|figura|tabella|Tab\.)\s+(\d+(?:\.\d+\w*)?)",
    # "espressione (3.1)" or "formula (6.9)"
    r"(?:espression[ei]|formula)\s+\((\d+\.\d+)\)",
    # EC2 style: "3.1.6 (1)P" or "10.3.1.1 (3)"
    r"(\d+(?:\.\d+){2,})\s*\(\d+\)\s*P?",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def extract_cross_references(text: str) -> list[str]:
    """Extract all cross-reference targets from a text chunk.

    Returns a deduplicated list of referenced section/clause numbers.
    """
    refs = set()
    for pattern in _COMPILED:
        for match in pattern.finditer(text):
            ref = match.group(1).strip().rstrip(".")
            refs.add(ref)
    return sorted(refs)
