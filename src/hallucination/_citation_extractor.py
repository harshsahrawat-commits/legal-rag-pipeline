"""Extract legal citations from LLM response text.

Pure regex extraction — no external dependencies.
Supports 7 citation pattern families for Indian legal documents.
"""

from __future__ import annotations

import re

from src.hallucination._models import CitationType, ExtractedCitation
from src.utils._logging import get_logger

_log = get_logger(__name__)

# --- Act alias mapping (superset of Phase 7's _ACT_ALIASES) ---

_ACT_ALIASES_EXTENDED: dict[str, str] = {
    "ipc": "Indian Penal Code",
    "crpc": "Code of Criminal Procedure",
    "cpc": "Code of Civil Procedure",
    "bns": "Bharatiya Nyaya Sanhita",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita",
    "bsa": "Bharatiya Sakshya Adhiniyam",
    "evidence act": "Indian Evidence Act",
    "contract act": "Indian Contract Act",
    "companies act": "Companies Act",
    "it act": "Information Technology Act",
    "negotiable instruments act": "Negotiable Instruments Act",
    "ndps act": "Narcotic Drugs and Psychotropic Substances Act",
    "poca": "Prevention of Corruption Act",
    "pocso": "Protection of Children from Sexual Offences Act",
    "domestic violence act": "Protection of Women from Domestic Violence Act",
    "rti act": "Right to Information Act",
    "sarfaesi act": "Securitisation and Reconstruction of Financial Assets and Enforcement of Securities Interest Act",
}


def resolve_act_alias(raw: str) -> str:
    """Resolve a short or colloquial act name to its full form."""
    cleaned = raw.strip().rstrip(",. ")
    return _ACT_ALIASES_EXTENDED.get(cleaned.lower(), cleaned)


# --- Compiled patterns ---

# 1. Section references: "Section 420 IPC", "S. 302 IPC", "Sec 10 Contract Act"
_SECTION_PATTERNS: list[re.Pattern[str]] = [
    # "Section 420 of the Indian Penal Code" — capture multi-word act name
    # Use non-greedy match that stops at punctuation or sentence boundaries
    re.compile(
        r"[Ss]ection\s+(\d+[A-Za-z]*)\s+of\s+(?:the\s+)?(.+?)(?=\s+(?:provides|deals|states|stipulates|prescribes|mandates|requires|reads|says|was|is|has|had|shall|may|can|under|where|which|who|and\s+[Ss]ection|and\s+[Aa]rticle|and\s+[Ss]ec)\b|[,;.)\n]|$)",
    ),
    # "S. 302 IPC" / "s.302 CrPC"
    re.compile(
        r"[Ss]\.\s*(\d+[A-Za-z]*)\s+([A-Z][A-Za-z]+)",
    ),
    # "Sec 10 Contract Act" / "Sec. 10 of Contract Act"
    re.compile(
        r"[Ss]ec\.?\s+(\d+[A-Za-z]*)\s+(?:of\s+(?:the\s+)?)?(.+?)(?=\s+(?:provides|deals|states|stipulates|prescribes|mandates|requires|reads|says|was|is|has|had|shall|may|can|under|where|which|who|and\s+[Ss]ection|and\s+[Aa]rticle|and\s+[Ss]ec)\b|[,;.)\n]|$)",
    ),
]

# 2. Article references: "Article 21", "Art. 14 of the Constitution"
_ARTICLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"[Aa]rticle\s+(\d+[A-Za-z]*)",
    ),
    re.compile(
        r"[Aa]rt\.\s*(\d+[A-Za-z]*)",
    ),
]

# 3. AIR citations: "AIR 2023 SC 1234"
_AIR_PATTERN = re.compile(
    r"AIR\s+(\d{4})\s+(\w+)\s+(\d+)",
)

# 4. SCC citations: "(2023) 5 SCC 678"
_SCC_PATTERN = re.compile(
    r"\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)",
)

# 5. SCC OnLine: "2023 SCC OnLine SC 890"
_SCC_ONLINE_PATTERN = re.compile(
    r"(\d{4})\s+SCC\s+OnLine\s+(\w+)\s+(\d+)",
)

# 6. Gazette notifications: "GSR 1234(E)", "S.O. 5678(E)"
_GAZETTE_PATTERN = re.compile(
    r"(?:GSR|S\.O\.)\s*(\d+)\s*\(E\)",
)

# 7. RBI/SEBI circulars: "RBI/2023-24/45", "SEBI/HO/2023/123"
_CIRCULAR_PATTERN = re.compile(
    r"((?:RBI|SEBI)[\w/.-]+\d+)",
)


def extract_citations(text: str) -> list[ExtractedCitation]:
    """Extract all legal citations from response text.

    Returns deduplicated citations in order of appearance.
    """
    citations: list[ExtractedCitation] = []
    seen_spans: set[tuple[int, int]] = set()

    def _add(citation: ExtractedCitation) -> None:
        span = (citation.span_start, citation.span_end)
        if span not in seen_spans:
            seen_spans.add(span)
            citations.append(citation)

    # --- Section references ---
    for pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(text):
            section = match.group(1).strip()
            act_raw = match.group(2).strip().rstrip(",. ")
            act = resolve_act_alias(act_raw)
            _add(
                ExtractedCitation(
                    text=match.group(0).strip(),
                    citation_type=CitationType.SECTION_REF,
                    section=section,
                    act=act,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )

    # --- Article references ---
    for pattern in _ARTICLE_PATTERNS:
        for match in pattern.finditer(text):
            article = match.group(1).strip()
            _add(
                ExtractedCitation(
                    text=match.group(0).strip(),
                    citation_type=CitationType.ARTICLE_REF,
                    article=article,
                    act="Constitution of India",
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )

    # --- AIR citations ---
    for match in _AIR_PATTERN.finditer(text):
        full = match.group(0).strip()
        _add(
            ExtractedCitation(
                text=full,
                citation_type=CitationType.CASE_CITATION,
                case_citation=full,
                span_start=match.start(),
                span_end=match.end(),
            )
        )

    # --- SCC citations ---
    for match in _SCC_PATTERN.finditer(text):
        full = match.group(0).strip()
        _add(
            ExtractedCitation(
                text=full,
                citation_type=CitationType.CASE_CITATION,
                case_citation=full,
                span_start=match.start(),
                span_end=match.end(),
            )
        )

    # --- SCC OnLine ---
    for match in _SCC_ONLINE_PATTERN.finditer(text):
        full = match.group(0).strip()
        _add(
            ExtractedCitation(
                text=full,
                citation_type=CitationType.CASE_CITATION,
                case_citation=full,
                span_start=match.start(),
                span_end=match.end(),
            )
        )

    # --- Gazette notifications ---
    for match in _GAZETTE_PATTERN.finditer(text):
        full = match.group(0).strip()
        _add(
            ExtractedCitation(
                text=full,
                citation_type=CitationType.NOTIFICATION_REF,
                notification_ref=full,
                span_start=match.start(),
                span_end=match.end(),
            )
        )

    # --- Circulars ---
    for match in _CIRCULAR_PATTERN.finditer(text):
        full = match.group(0).strip()
        # Skip if already captured as part of another citation
        if any(c.span_start <= match.start() < c.span_end for c in citations):
            continue
        _add(
            ExtractedCitation(
                text=full,
                citation_type=CitationType.CIRCULAR_REF,
                circular_ref=full,
                span_start=match.start(),
                span_end=match.end(),
            )
        )

    # Sort by position
    citations.sort(key=lambda c: c.span_start)
    return citations
