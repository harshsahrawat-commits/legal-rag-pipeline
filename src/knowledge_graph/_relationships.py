"""Relationship builder for the knowledge graph.

Builds typed relationships between extracted entities from LegalChunk metadata.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.knowledge_graph._models import Relationship
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.chunking._models import LegalChunk
    from src.knowledge_graph._models import ExtractedEntities

_log = get_logger(__name__)

# Pattern: "Section 302, IPC" or "Section 302 of the Indian Penal Code"
_SECTION_REF_PATTERN = re.compile(
    r"Section\s+(\d+[A-Za-z]*)"
    r"(?:\s*,\s*|\s+of\s+(?:the\s+)?)"
    r"(.+?)$",
    re.IGNORECASE,
)

# Common act abbreviations
_ACT_ABBREVS: dict[str, str] = {
    "IPC": "Indian Penal Code",
    "CrPC": "Code of Criminal Procedure",
    "CPC": "Code of Civil Procedure",
    "IEA": "Indian Evidence Act",
    "IT Act": "Information Technology Act",
    "BNS": "Bharatiya Nyaya Sanhita",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
    "BSA": "Bharatiya Sakshya Adhiniyam",
}


class RelationshipBuilder:
    """Build relationships between knowledge graph entities."""

    def build_from_chunk(
        self, chunk: LegalChunk, entities: ExtractedEntities
    ) -> list[Relationship]:
        """Route relationship building based on document type."""
        from src.acquisition._models import DocumentType

        if chunk.document_type == DocumentType.STATUTE:
            return self.build_statute_relationships(chunk, entities)
        if chunk.document_type == DocumentType.JUDGMENT:
            return self.build_judgment_relationships(chunk, entities)
        return []

    def build_statute_relationships(
        self, chunk: LegalChunk, entities: ExtractedEntities
    ) -> list[Relationship]:
        """Build relationships for statute chunks."""
        rels: list[Relationship] = []

        if not chunk.statute:
            return rels

        meta = chunk.statute

        # Act -[:CONTAINS]-> Section
        for section in entities.sections:
            rels.append(
                Relationship(
                    from_label="Act",
                    from_key={"name": meta.act_name},
                    to_label="Section",
                    to_key={"parent_act": section.parent_act, "number": section.number},
                    rel_type="CONTAINS",
                )
            )

        # Section -[:HAS_VERSION]-> SectionVersion
        for sv in entities.section_versions:
            if meta.section_number:
                rels.append(
                    Relationship(
                        from_label="Section",
                        from_key={"parent_act": meta.act_name, "number": meta.section_number},
                        to_label="SectionVersion",
                        to_key={"version_id": sv.version_id},
                        rel_type="HAS_VERSION",
                    )
                )

        # Amendment relationships
        for amendment in entities.amendments:
            if meta.section_number:
                rel_type = self._amendment_rel_type(amendment.nature)
                rels.append(
                    Relationship(
                        from_label="Amendment",
                        from_key={
                            "amending_act": amendment.amending_act,
                            "date": str(amendment.date),
                            "nature": amendment.nature,
                        },
                        to_label="Section",
                        to_key={"parent_act": meta.act_name, "number": meta.section_number},
                        rel_type=rel_type,
                    )
                )

        # Act -[:REPEALS]-> Act (if repealed_by is set)
        if meta.repealed_by:
            rels.append(
                Relationship(
                    from_label="Act",
                    from_key={"name": meta.repealed_by},
                    to_label="Act",
                    to_key={"name": meta.act_name},
                    rel_type="REPEALS",
                )
            )

        # Section -[:REFERENCES]-> Section (cross-references within statutes)
        for ref_str in chunk.content.sections_cited:
            parsed = self._parse_section_ref(ref_str)
            if parsed and meta.section_number:
                ref_act, ref_num = parsed
                # Skip self-references
                if ref_act == meta.act_name and ref_num == meta.section_number:
                    continue
                rels.append(
                    Relationship(
                        from_label="Section",
                        from_key={"parent_act": meta.act_name, "number": meta.section_number},
                        to_label="Section",
                        to_key={"parent_act": ref_act, "number": ref_num},
                        rel_type="REFERENCES",
                    )
                )

        # Section -[:DEFINES]-> LegalConcept
        for concept in entities.legal_concepts:
            if meta.section_number:
                rels.append(
                    Relationship(
                        from_label="Section",
                        from_key={"parent_act": meta.act_name, "number": meta.section_number},
                        to_label="LegalConcept",
                        to_key={"name": concept.name},
                        rel_type="DEFINES",
                    )
                )

        return rels

    def build_judgment_relationships(
        self, chunk: LegalChunk, entities: ExtractedEntities
    ) -> list[Relationship]:
        """Build relationships for judgment chunks."""
        rels: list[Relationship] = []

        if not chunk.judgment:
            return rels

        meta = chunk.judgment

        # Judgment -[:FILED_IN]-> Court
        for court in entities.courts:
            rels.append(
                Relationship(
                    from_label="Judgment",
                    from_key={"citation": meta.case_citation},
                    to_label="Court",
                    to_key={"name": court.name},
                    rel_type="FILED_IN",
                )
            )

        # Judgment -[:DECIDED_BY]-> Judge
        for judge in entities.judges:
            rels.append(
                Relationship(
                    from_label="Judgment",
                    from_key={"citation": meta.case_citation},
                    to_label="Judge",
                    to_key={"name": judge.name},
                    rel_type="DECIDED_BY",
                )
            )

        # Judgment -[:INTERPRETS / CITES_SECTION]-> Section
        for ref_str in chunk.content.sections_cited:
            parsed = self._parse_section_ref(ref_str)
            if parsed:
                ref_act, ref_num = parsed
                rels.append(
                    Relationship(
                        from_label="Judgment",
                        from_key={"citation": meta.case_citation},
                        to_label="Section",
                        to_key={"parent_act": ref_act, "number": ref_num},
                        rel_type="CITES_SECTION",
                    )
                )
                rels.append(
                    Relationship(
                        from_label="Judgment",
                        from_key={"citation": meta.case_citation},
                        to_label="Section",
                        to_key={"parent_act": ref_act, "number": ref_num},
                        rel_type="INTERPRETS",
                    )
                )

        # Judgment -[:CITES_CASE]-> Judgment
        for case_ref in chunk.content.cases_cited:
            normalized = self._normalize_citation(case_ref)
            if normalized and normalized != meta.case_citation:
                rels.append(
                    Relationship(
                        from_label="Judgment",
                        from_key={"citation": meta.case_citation},
                        to_label="Judgment",
                        to_key={"citation": normalized},
                        rel_type="CITES_CASE",
                    )
                )

        # Judgment -[:OVERRULES]-> Judgment (reverse: overruled_by means this was overruled)
        if meta.overruled_by:
            rels.append(
                Relationship(
                    from_label="Judgment",
                    from_key={"citation": meta.overruled_by},
                    to_label="Judgment",
                    to_key={"citation": meta.case_citation},
                    rel_type="OVERRULES",
                )
            )

        # Judgment -[:FOLLOWS]-> Judgment (reverse: followed_in means others followed this)
        for follower in meta.followed_in:
            rels.append(
                Relationship(
                    from_label="Judgment",
                    from_key={"citation": follower},
                    to_label="Judgment",
                    to_key={"citation": meta.case_citation},
                    rel_type="FOLLOWS",
                )
            )

        # Judgment -[:DISTINGUISHES]-> Judgment (reverse: distinguished_in)
        for distinguisher in meta.distinguished_in:
            rels.append(
                Relationship(
                    from_label="Judgment",
                    from_key={"citation": distinguisher},
                    to_label="Judgment",
                    to_key={"citation": meta.case_citation},
                    rel_type="DISTINGUISHES",
                )
            )

        return rels

    @staticmethod
    def _amendment_rel_type(nature: str) -> str:
        """Map amendment nature to relationship type."""
        nature_lower = nature.lower()
        if "insert" in nature_lower:
            return "INSERTS"
        if "omis" in nature_lower or "omit" in nature_lower or "repeal" in nature_lower:
            return "OMITS"
        return "AMENDS"

    @staticmethod
    def _parse_section_ref(ref: str) -> tuple[str, str] | None:
        """Parse 'Section 302, IPC' into ('Indian Penal Code', '302')."""
        match = _SECTION_REF_PATTERN.match(ref.strip())
        if not match:
            return None
        section_num = match.group(1)
        act_ref = match.group(2).strip().rstrip(".")
        # Expand abbreviation
        act_name = _ACT_ABBREVS.get(act_ref, act_ref)
        return act_name, section_num

    @staticmethod
    def _normalize_citation(citation: str) -> str | None:
        """Normalize a case citation string."""
        citation = citation.strip()
        if not citation:
            return None
        return citation
