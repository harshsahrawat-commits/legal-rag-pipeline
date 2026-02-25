from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, Field

from src.acquisition._models import SourceType  # noqa: TC001

# --- Node models ---


class ActNode(BaseModel):
    """An Act (statute) in the knowledge graph."""

    name: str
    number: str | None = None
    year: int | None = None
    date_enacted: date | None = None
    date_effective: date | None = None
    date_repealed: date | None = None
    jurisdiction: str = "India"
    status: str = "in_force"


class SectionNode(BaseModel):
    """A section within an Act."""

    number: str
    parent_act: str
    chapter: str | None = None
    part: str | None = None
    is_in_force: bool = True
    chunk_id: UUID | None = None


class SectionVersionNode(BaseModel):
    """A specific version of a section's text at a point in time."""

    version_id: str
    text_hash: str
    effective_from: date | None = None
    effective_until: date | None = None
    amending_act: str | None = None


class JudgmentNode(BaseModel):
    """A court judgment in the knowledge graph."""

    citation: str
    alt_citations: list[str] = Field(default_factory=list)
    court: str
    court_level: int
    bench_type: str | None = None
    bench_strength: int | None = None
    date_decided: date | None = None
    case_type: str | None = None
    parties_petitioner: str | None = None
    parties_respondent: str | None = None
    status: str = "good_law"
    chunk_id: UUID | None = None


class AmendmentNode(BaseModel):
    """An amendment that modifies one or more sections."""

    amending_act: str
    date: date
    gazette_ref: str | None = None
    nature: str


class LegalConceptNode(BaseModel):
    """A defined legal concept/term."""

    name: str
    definition_source: str | None = None
    category: str | None = None


class CourtNode(BaseModel):
    """A court in the Indian judicial system."""

    name: str
    hierarchy_level: int
    state: str | None = None
    jurisdiction_type: str | None = None


class JudgeNode(BaseModel):
    """A judge who decided cases."""

    name: str
    courts_served: list[str] = Field(default_factory=list)


# --- Relationship model ---


class Relationship(BaseModel):
    """A relationship to MERGE between two nodes."""

    from_label: str
    from_key: dict[str, object]
    to_label: str
    to_key: dict[str, object]
    rel_type: str
    properties: dict[str, object] = Field(default_factory=dict)


# --- Extraction result model ---


class ExtractedEntities(BaseModel):
    """All entities extracted from a single chunk."""

    acts: list[ActNode] = Field(default_factory=list)
    sections: list[SectionNode] = Field(default_factory=list)
    section_versions: list[SectionVersionNode] = Field(default_factory=list)
    judgments: list[JudgmentNode] = Field(default_factory=list)
    amendments: list[AmendmentNode] = Field(default_factory=list)
    legal_concepts: list[LegalConceptNode] = Field(default_factory=list)
    courts: list[CourtNode] = Field(default_factory=list)
    judges: list[JudgeNode] = Field(default_factory=list)


# --- Integrity models ---


class IntegrityCheck(BaseModel):
    """Result of a single integrity check."""

    name: str
    passed: bool
    violations: list[str] = Field(default_factory=list)


class IntegrityReport(BaseModel):
    """Aggregate result of all integrity checks."""

    passed: bool
    checks: list[IntegrityCheck] = Field(default_factory=list)


# --- Config models ---


class KGSettings(BaseModel):
    """Knowledge graph settings from configs/knowledge_graph.yaml."""

    input_dir: Path = Path("data/enriched")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    batch_size: int = 100


class KGConfig(BaseModel):
    """Root model for configs/knowledge_graph.yaml."""

    settings: KGSettings = Field(default_factory=KGSettings)


# --- Pipeline result model ---


class KGResult(BaseModel):
    """Summary of a knowledge graph pipeline run."""

    source_type: SourceType | None = None
    documents_found: int = 0
    documents_ingested: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    integrity_passed: bool | None = None
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
