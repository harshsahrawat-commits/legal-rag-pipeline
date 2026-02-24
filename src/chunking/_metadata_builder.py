"""Centralizes metadata construction for all chunkers.

Builds ``SourceInfo``, ``StatuteMetadata``, ``JudgmentMetadata``,
``ContentMetadata``, and ``IngestionMetadata`` from a ``ParsedDocument``.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from src.chunking._models import (
    ChunkStrategy,
    ChunkType,
    ContentMetadata,
    CourtHierarchy,
    IngestionMetadata,
    JudgmentMetadata,
    SourceInfo,
    StatuteMetadata,
)
from src.parsing._models import SectionLevel

if TYPE_CHECKING:
    from src.parsing._models import ParsedDocument

# --- Regex patterns for citation extraction ---

_SECTION_CITED_RE = re.compile(
    r"(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\.\d+)?(?:\s*\(\d+\))?)"
    r"(?:\s+(?:of|,)\s+(?:the\s+)?([A-Z][A-Za-z\s,]+(?:Act|Code)[^.;]*?))?",
)

_ACT_CITED_RE = re.compile(
    r"(?:the\s+)?((?:[A-Z][A-Za-z]+\s+){1,6}(?:Act|Code|Ordinance),?\s*\d{4})",
)

_CASE_CITED_RE = re.compile(
    r"(?:AIR\s+\d{4}\s+\w+\s+\d+|\(\d{4}\)\s+\d+\s+SCC\s+\d+)",
)

# --- Court classification ---

_COURT_KEYWORDS: list[tuple[str, CourtHierarchy]] = [
    ("supreme court", CourtHierarchy.SUPREME_COURT),
    ("high court", CourtHierarchy.HIGH_COURT),
    ("district court", CourtHierarchy.DISTRICT_COURT),
    ("sessions court", CourtHierarchy.DISTRICT_COURT),
    ("tribunal", CourtHierarchy.TRIBUNAL),
    ("commission", CourtHierarchy.QUASI_JUDICIAL),
    ("authority", CourtHierarchy.QUASI_JUDICIAL),
]

# --- SectionLevel → ChunkType mapping ---

_LEVEL_TO_CHUNK_TYPE: dict[SectionLevel, ChunkType] = {
    SectionLevel.PREAMBLE: ChunkType.STATUTORY_TEXT,
    SectionLevel.PART: ChunkType.STATUTORY_TEXT,
    SectionLevel.CHAPTER: ChunkType.STATUTORY_TEXT,
    SectionLevel.SECTION: ChunkType.STATUTORY_TEXT,
    SectionLevel.SUB_SECTION: ChunkType.STATUTORY_TEXT,
    SectionLevel.CLAUSE: ChunkType.STATUTORY_TEXT,
    SectionLevel.SUB_CLAUSE: ChunkType.STATUTORY_TEXT,
    SectionLevel.PROVISO: ChunkType.PROVISO,
    SectionLevel.EXPLANATION: ChunkType.EXPLANATION,
    SectionLevel.DEFINITION: ChunkType.DEFINITION,
    SectionLevel.SCHEDULE: ChunkType.SCHEDULE_ENTRY,
    SectionLevel.HEADER: ChunkType.STATUTORY_TEXT,
    SectionLevel.FACTS: ChunkType.FACTS,
    SectionLevel.ISSUES: ChunkType.ISSUES,
    SectionLevel.REASONING: ChunkType.REASONING,
    SectionLevel.HOLDING: ChunkType.HOLDING,
    SectionLevel.ORDER: ChunkType.ORDER,
    SectionLevel.DISSENT: ChunkType.DISSENT,
    SectionLevel.OBITER: ChunkType.OBITER,
    SectionLevel.PARAGRAPH: ChunkType.STATUTORY_TEXT,
}


class MetadataBuilder:
    """Constructs chunk metadata sub-models from a ``ParsedDocument``."""

    def build_source_info(self, doc: ParsedDocument) -> SourceInfo:
        """Build provenance information from the parsed document."""
        return SourceInfo(
            url=doc.raw_content_path,
            source_name=doc.source_type.value,
            scraped_at=doc.parsed_at,
            last_verified=doc.parsed_at,
        )

    def build_statute_metadata(
        self,
        doc: ParsedDocument,
        *,
        section_number: str | None = None,
        chapter: str | None = None,
        part: str | None = None,
        schedule: str | None = None,
    ) -> StatuteMetadata:
        """Build statute-specific metadata."""
        date_enacted = _parse_date(doc.date) if doc.date else None
        return StatuteMetadata(
            act_name=doc.act_name or doc.title or "Unknown Act",
            act_number=doc.act_number,
            section_number=section_number,
            chapter=chapter,
            part=part,
            schedule=schedule,
            date_enacted=date_enacted,
        )

    def build_judgment_metadata(
        self,
        doc: ParsedDocument,
    ) -> JudgmentMetadata:
        """Build judgment-specific metadata."""
        court = doc.court or "Unknown Court"
        court_level = self.classify_court_hierarchy(court)

        petitioner = None
        respondent = None
        if doc.parties:
            parts = re.split(
                r"\s+(?:vs?\.?|versus)\s+", doc.parties, maxsplit=1, flags=re.IGNORECASE
            )
            if len(parts) == 2:
                petitioner = parts[0].strip()
                respondent = parts[1].strip()

        return JudgmentMetadata(
            case_citation=doc.case_citation or "Unknown Citation",
            court=court,
            court_level=court_level,
            date_decided=_parse_date(doc.date) if doc.date else None,
            parties_petitioner=petitioner,
            parties_respondent=respondent,
        )

    def build_content_metadata(self, text: str) -> ContentMetadata:
        """Extract cited sections, acts, and cases from chunk text."""
        sections_cited: list[str] = []
        for m in _SECTION_CITED_RE.finditer(text):
            section_ref = f"Section {m.group(1)}"
            if m.group(2):
                section_ref += f" of {m.group(2).strip().rstrip(',')}"
            if section_ref not in sections_cited:
                sections_cited.append(section_ref)

        acts_cited = list(dict.fromkeys(m.group(1).strip() for m in _ACT_CITED_RE.finditer(text)))
        cases_cited = list(dict.fromkeys(m.group(0) for m in _CASE_CITED_RE.finditer(text)))

        has_hindi = bool(re.search(r"[\u0900-\u097F]", text))

        return ContentMetadata(
            sections_cited=sections_cited,
            acts_cited=acts_cited,
            cases_cited=cases_cited,
            language="hi" if has_hindi and len(re.findall(r"[\u0900-\u097F]", text)) > 20 else "en",
            has_hindi=has_hindi,
        )

    def build_ingestion_metadata(
        self,
        doc: ParsedDocument,
        chunk_strategy: ChunkStrategy,
        *,
        requires_manual_review: bool = False,
    ) -> IngestionMetadata:
        """Build processing provenance metadata."""
        return IngestionMetadata(
            ingested_at=datetime.now(UTC),
            parser=doc.parser_used.value,
            ocr_confidence=doc.ocr_confidence,
            quality_score=doc.quality.overall_score,
            chunk_strategy=chunk_strategy,
            requires_manual_review=requires_manual_review,
        )

    @staticmethod
    def classify_chunk_type(level: SectionLevel) -> ChunkType:
        """Map a ``SectionLevel`` to the appropriate ``ChunkType``."""
        return _LEVEL_TO_CHUNK_TYPE.get(level, ChunkType.STATUTORY_TEXT)

    @staticmethod
    def classify_court_hierarchy(court: str) -> CourtHierarchy:
        """Classify a court name string into the hierarchy."""
        court_lower = court.lower()
        for keyword, hierarchy in _COURT_KEYWORDS:
            if keyword in court_lower:
                return hierarchy
        return CourtHierarchy.DISTRICT_COURT


def _parse_date(date_str: str) -> date | None:
    """Best-effort date parsing from Indian legal date formats."""
    for fmt in ("%d %B, %Y", "%d-%b-%Y", "%d %B %Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None
