"""Entity extraction from LegalChunk metadata.

Extracts Act, Section, SectionVersion, Judgment, Amendment, LegalConcept,
Court, and Judge entities from chunk metadata fields.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from src.knowledge_graph._models import (
    ActNode,
    AmendmentNode,
    CourtNode,
    ExtractedEntities,
    JudgeNode,
    JudgmentNode,
    LegalConceptNode,
    SectionNode,
    SectionVersionNode,
)
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk

_log = get_logger(__name__)

_YEAR_PATTERN = re.compile(r"(?:of\s+)?(\d{4})")


class EntityExtractor:
    """Extract knowledge graph entities from LegalChunk metadata."""

    def extract_from_chunk(self, chunk: LegalChunk) -> ExtractedEntities:
        """Route extraction based on document type."""
        from src.acquisition._models import DocumentType

        if chunk.document_type == DocumentType.STATUTE:
            return self.extract_from_statute_chunk(chunk)
        if chunk.document_type == DocumentType.JUDGMENT:
            return self.extract_from_judgment_chunk(chunk)
        return ExtractedEntities()

    def extract_from_statute_chunk(self, chunk: LegalChunk) -> ExtractedEntities:
        """Extract entities from a statute chunk."""
        entities = ExtractedEntities()

        if chunk.statute is None:
            return entities

        meta = chunk.statute

        # Act node
        act = ActNode(
            name=meta.act_name,
            number=meta.act_number,
            year=self._extract_year(meta.act_number, meta.date_enacted),
            date_enacted=meta.date_enacted,
            date_effective=meta.date_effective,
            date_repealed=meta.date_repealed,
            status=self._determine_act_status(meta),
        )
        entities.acts.append(act)

        # Section node
        if meta.section_number:
            section = SectionNode(
                number=meta.section_number,
                parent_act=meta.act_name,
                chapter=meta.chapter,
                part=meta.part,
                is_in_force=meta.is_in_force,
                chunk_id=chunk.id,
            )
            entities.sections.append(section)

            # SectionVersion node
            text_hash = self._compute_text_hash(chunk.text)
            version_id = f"{meta.act_name}:{meta.section_number}:{text_hash[:8]}"
            sv = SectionVersionNode(
                version_id=version_id,
                text_hash=text_hash,
                effective_from=meta.date_effective or meta.date_enacted,
            )
            entities.section_versions.append(sv)

        # Amendment nodes
        for record in meta.amendment_history:
            amendment = AmendmentNode(
                amending_act=record.amending_act,
                date=record.date,
                gazette_ref=record.gazette_ref,
                nature=record.nature,
            )
            entities.amendments.append(amendment)

        # Legal concepts from definition chunks
        entities.legal_concepts.extend(self.extract_legal_concepts(chunk))

        return entities

    def extract_from_judgment_chunk(self, chunk: LegalChunk) -> ExtractedEntities:
        """Extract entities from a judgment chunk."""
        entities = ExtractedEntities()

        if chunk.judgment is None:
            return entities

        meta = chunk.judgment

        # Judgment node
        judgment = JudgmentNode(
            citation=meta.case_citation,
            alt_citations=meta.alt_citations,
            court=meta.court,
            court_level=int(meta.court_level),
            bench_type=meta.bench_type,
            bench_strength=meta.bench_strength,
            date_decided=meta.date_decided,
            case_type=meta.case_type,
            parties_petitioner=meta.parties_petitioner,
            parties_respondent=meta.parties_respondent,
            status=self._determine_judgment_status(meta),
            chunk_id=chunk.id,
        )
        entities.judgments.append(judgment)

        # Court node
        court = CourtNode(
            name=meta.court,
            hierarchy_level=int(meta.court_level),
        )
        entities.courts.append(court)

        # Judge nodes
        for name in meta.judge_names:
            judge = JudgeNode(
                name=name,
                courts_served=[meta.court],
            )
            entities.judges.append(judge)

        return entities

    def extract_legal_concepts(self, chunk: LegalChunk) -> list[LegalConceptNode]:
        """Extract LegalConcept nodes from definition chunks."""
        from src.chunking._models import ChunkType

        if chunk.chunk_type != ChunkType.DEFINITION:
            return []

        concepts = []
        for concept_name in chunk.content.legal_concepts:
            source = None
            if chunk.statute and chunk.statute.section_number:
                source = f"Section {chunk.statute.section_number}, {chunk.statute.act_name}"
            concepts.append(
                LegalConceptNode(
                    name=concept_name,
                    definition_source=source,
                )
            )
        return concepts

    @staticmethod
    def _extract_year(act_number: str | None, date_enacted: object | None) -> int | None:
        """Extract year from act number like 'Act No. 45 of 1860' or from date."""
        if act_number:
            match = _YEAR_PATTERN.search(act_number)
            if match:
                return int(match.group(1))
        if date_enacted is not None and hasattr(date_enacted, "year"):
            return date_enacted.year
        return None

    @staticmethod
    def _compute_text_hash(text: str) -> str:
        """SHA256 hash of the text for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _determine_act_status(meta: object) -> str:
        """Determine Act status from StatuteMetadata."""
        from src.chunking._models import TemporalStatus

        ts = meta.temporal_status
        if ts == TemporalStatus.REPEALED:
            return "repealed"
        if ts == TemporalStatus.PARTIALLY_REPEALED:
            return "partially_repealed"
        return "in_force"

    @staticmethod
    def _determine_judgment_status(meta: object) -> str:
        """Determine Judgment status from JudgmentMetadata."""
        if meta.is_overruled:
            return "overruled"
        if meta.distinguished_in:
            return "distinguished"
        return "good_law"
