# Phase 6: Knowledge Graph — Implementation Plan

## Context

Phases 1-5 (Acquisition → Embedding & Indexing) are complete with 827 tests. Phase 5 outputs enriched chunks indexed in Qdrant (dual vectors + BM25 sparse) and parent text in Redis. Phase 6 builds the Neo4j knowledge graph that enables point-in-time retrieval, citation traversal, amendment cascading, hierarchy navigation, and direct KG query answering for SIMPLE queries.

The KG is consumed by:
- **Phase 7 (Retrieval):** Graph traversal retrieves related chunks alongside vector search
- **Phase 8 (Hallucination Mitigation):** Citation verification + temporal consistency checks query the KG
- **Phase 0 (Query Intelligence):** SIMPLE queries are answered by KG alone

## Scope

### In Scope (Phase 6)

1. **Neo4j Schema Setup** — Create all node labels, relationship types, constraints, and indexes per `docs/knowledge_graph_schema.md`
2. **Entity Extractors** — Extract Act, Section, SectionVersion, Judgment, Amendment, LegalConcept, Court, and Judge entities from `LegalChunk` metadata
3. **Relationship Builders** — Build all relationship types: CONTAINS, HAS_VERSION, AMENDS/INSERTS/OMITS, REPEALS/REPLACES, INTERPRETS, CITES_SECTION, CITES_CASE, OVERRULES/DISTINGUISHES/FOLLOWS, DECIDED_BY, FILED_IN, REFERENCES, DEFINES
4. **KG Ingestion Pipeline** — Orchestrator that reads enriched chunks from `data/enriched/`, extracts entities + relationships, and upserts into Neo4j
5. **Query Builders** — Reusable Cypher query functions for the 4 critical queries: point-in-time retrieval, amendment cascading, citation traversal, hierarchy navigation
6. **Integrity Checker** — Post-ingestion validation: every Section has a SectionVersion, repealed Acts have all sections `is_in_force=false`, OVERRULES hierarchy check, no overlapping SectionVersion date ranges
7. **CLI** — Command-line interface consistent with previous phases

### Out of Scope (Deferred)

- Full-text search in Neo4j (vector search is in Qdrant)
- Graph-based reranking (Phase 7)
- Citation verification logic (Phase 8 — but query builders from Phase 6 are its foundation)

## Data Flow

```
Input:
  data/enriched/{source}/{doc_id}.json     — list[LegalChunk] (from Phase 5)

Processing:
  Step 1: EntityExtractor     — LegalChunk metadata → node dicts
  Step 2: RelationshipBuilder — Cross-chunk/cross-doc relationship extraction
  Step 3: Neo4jClient         — Cypher MERGE upserts (idempotent)
  Step 4: IntegrityChecker    — Post-ingestion validation

Output:
  Neo4j graph database with 8 node types and 15 relationship types
  Integrity report (pass/fail with details)
```

## Module Structure

```
src/knowledge_graph/
├── __init__.py              # Public API: KnowledgeGraphConfig, KnowledgeGraphPipeline, KGResult
├── __main__.py              # Module runner
├── _config.py               # YAML config loader (load_kg_config)
├── _exceptions.py           # KnowledgeGraphError hierarchy
├── _models.py               # Pydantic models: KGSettings, KGConfig, KGResult, node/rel models
├── _client.py               # Neo4jClient: connection, schema setup, MERGE operations, queries
├── _extractors.py           # EntityExtractor: LegalChunk → node models
├── _relationships.py        # RelationshipBuilder: cross-entity relationship construction
├── _queries.py              # QueryBuilder: reusable Cypher for point-in-time, traversal, etc.
├── _integrity.py            # IntegrityChecker: post-ingestion validation
├── pipeline.py              # KnowledgeGraphPipeline orchestrator
└── run.py                   # CLI entry point

configs/knowledge_graph.yaml  # KG config
tests/knowledge_graph/        # ~170 tests
```

## Key Design Decisions

1. **Neo4j driver: lazy import** — `neo4j` package is optional. Lazy import in `_client.py` via `_ensure_driver()`, matching the pattern from `_qdrant_indexer.py`. Raises `KGNotAvailableError` if not installed.

2. **MERGE-based upserts (idempotent)** — All writes use Cypher `MERGE` with `ON CREATE SET` / `ON MATCH SET` clauses. Re-running the pipeline on the same input produces the same graph state. This follows the project's idempotency rule.

3. **Batch transactions** — Group multiple MERGE operations per transaction (configurable `batch_size`, default 100). One transaction per document's entities, separate transaction for cross-document relationships.

4. **Entity models as Pydantic** — Each node type has a Pydantic model (`ActNode`, `SectionNode`, `JudgmentNode`, etc.) for validated data between extraction and Cypher generation. This follows the "never raw dicts for structured data" convention.

5. **Chunk-driven extraction** — Entities are extracted from `LegalChunk` metadata fields, NOT by re-parsing raw text. `StatuteMetadata` → Act + Section + SectionVersion + Amendment nodes. `JudgmentMetadata` → Judgment + Court + Judge nodes. `ContentMetadata` → CITES_SECTION/CITES_CASE/LegalConcept relationships.

6. **Cross-document relationships** — Some relationships (CITES_CASE, INTERPRETS, OVERRULES) reference entities from other documents. These are built as "dangling" MERGE operations — if the target node doesn't exist yet, MERGE creates a stub. When the target document is later ingested, the stub is enriched with full properties via `ON MATCH SET`.

7. **SectionVersion tracking** — Each Section node gets a SectionVersion for every distinct text version. `effective_from` comes from `StatuteMetadata.date_effective` or `AmendmentRecord.date`. `effective_until` is set when a subsequent amendment is ingested. `text_hash` = SHA256 of chunk text for deduplication.

8. **Async driver** — Use `neo4j.AsyncGraphDatabase.driver()` for consistency with the project's async pipeline pattern.

## Node & Relationship Models

### Node Models (Pydantic)

```python
class ActNode(BaseModel):
    name: str                    # From StatuteMetadata.act_name
    number: str | None           # From StatuteMetadata.act_number
    year: int | None             # Extracted from act_number or date_enacted
    date_enacted: date | None
    date_effective: date | None
    date_repealed: date | None
    jurisdiction: str = "India"
    status: str                  # in_force / repealed / partially_repealed

class SectionNode(BaseModel):
    number: str                  # From StatuteMetadata.section_number
    parent_act: str              # From StatuteMetadata.act_name
    chapter: str | None
    part: str | None
    is_in_force: bool
    chunk_id: UUID               # Link back to Qdrant

class SectionVersionNode(BaseModel):
    version_id: str              # "{parent_act}:{number}:v{n}" or hash-based
    text_hash: str               # SHA256 of section text
    effective_from: date | None
    effective_until: date | None
    amending_act: str | None

class JudgmentNode(BaseModel):
    citation: str                # Primary citation
    alt_citations: list[str]
    court: str
    court_level: int             # CourtHierarchy int value
    bench_type: str | None
    bench_strength: int | None
    date_decided: date | None
    case_type: str | None
    parties_petitioner: str | None
    parties_respondent: str | None
    status: str                  # good_law / overruled / distinguished
    chunk_id: UUID               # Link back to Qdrant header chunk

class AmendmentNode(BaseModel):
    amending_act: str
    date: date
    gazette_ref: str | None
    nature: str                  # substitution / insertion / omission

class LegalConceptNode(BaseModel):
    name: str
    definition_source: str | None  # "Section 2, IPC" etc.
    category: str | None

class CourtNode(BaseModel):
    name: str
    hierarchy_level: int
    state: str | None
    jurisdiction_type: str | None  # "constitutional" / "criminal" / "civil" etc.

class JudgeNode(BaseModel):
    name: str
    courts_served: list[str]
```

### Relationship Extraction Logic

| Relationship | Source | Extraction Method |
|---|---|---|
| `(Act)-[:CONTAINS]->(Section)` | StatuteMetadata | Every statute chunk creates Act→Section |
| `(Section)-[:HAS_VERSION]->(SectionVersion)` | Chunk text + amendment_history | SHA256 of text, dates from metadata |
| `(Amendment)-[:AMENDS]->(Section)` | StatuteMetadata.amendment_history | One Amendment node per AmendmentRecord |
| `(Amendment)-[:INSERTS]->(Section)` | AmendmentRecord.nature == "insertion" | |
| `(Amendment)-[:OMITS]->(Section)` | AmendmentRecord.nature == "omission" | |
| `(Act)-[:REPEALS]->(Act)` | StatuteMetadata.repealed_by | Creates stub Act if target not yet ingested |
| `(Act)-[:REPLACES]->(Act)` | StatuteMetadata.replaced_by_section | Inferred when new Act covers same topic |
| `(Judgment)-[:INTERPRETS]->(Section)` | ContentMetadata.sections_cited | For judgments that cite statute sections |
| `(Judgment)-[:CITES_SECTION]->(Section)` | ContentMetadata.sections_cited | All section references |
| `(Judgment)-[:CITES_CASE]->(Judgment)` | ContentMetadata.cases_cited | Stub Judgment if target not yet ingested |
| `(Judgment)-[:OVERRULES]->(Judgment)` | JudgmentMetadata.overruled_by (reverse) | |
| `(Judgment)-[:DISTINGUISHES]->(Judgment)` | JudgmentMetadata.distinguished_in (reverse) | |
| `(Judgment)-[:FOLLOWS]->(Judgment)` | JudgmentMetadata.followed_in (reverse) | |
| `(Judgment)-[:DECIDED_BY]->(Judge)` | JudgmentMetadata.judge_names | One per judge |
| `(Judgment)-[:FILED_IN]->(Court)` | JudgmentMetadata.court | |
| `(Section)-[:REFERENCES]->(Section)` | ContentMetadata.sections_cited (for statutes) | Statute cross-references |
| `(Section)-[:DEFINES]->(LegalConcept)` | ContentMetadata.legal_concepts (for definitions) | ChunkType.DEFINITION chunks |

## Critical Queries (QueryBuilder)

These are reusable Cypher query functions used by Phase 7 (Retrieval) and Phase 8 (Hallucination Mitigation):

```python
class QueryBuilder:
    async def point_in_time(self, act: str, section: str, query_date: date) -> SectionVersionNode | None:
        """What was Section X of Act Y as on a given date?"""

    async def amendment_cascade(self, amending_act: str) -> list[dict]:
        """Find all sections affected by an amendment."""

    async def citation_traversal(self, section: str, act: str, court: str | None = None) -> list[JudgmentNode]:
        """Find all judgments interpreting/citing a section, optionally filtered by court."""

    async def hierarchy_navigation(self, act: str, chapter: str | None = None) -> list[SectionNode]:
        """List all sections under an Act, optionally filtered by chapter."""

    async def temporal_status(self, section: str, act: str, ref_date: date | None = None) -> dict:
        """Check if a section is currently in force, repealed, or superseded."""

    async def judgment_relationships(self, citation: str) -> dict:
        """Get all relationships for a judgment (overruled by, follows, distinguishes)."""

    async def find_replacement(self, old_act: str, section: str) -> dict | None:
        """Find the replacement section/act for a repealed provision."""

    async def node_exists(self, identifier: str) -> bool:
        """Check if an Act, Section, or Judgment node exists (for citation verification)."""
```

## Subtask Breakdown

### Subtask 1: Foundation — Exceptions, Models, Config (~30 tests)

**Files:**
- `src/knowledge_graph/_exceptions.py` — `KnowledgeGraphError(LegalRAGError)`, `KGConnectionError`, `KGSchemaError`, `KGIngestionError`, `KGQueryError`, `KGIntegrityError`, `KGNotAvailableError`
- `src/knowledge_graph/_models.py` — All Pydantic node models (ActNode, SectionNode, SectionVersionNode, JudgmentNode, AmendmentNode, LegalConceptNode, CourtNode, JudgeNode) + `KGSettings`, `KGConfig`, `KGResult`
- `src/knowledge_graph/_config.py` — `load_kg_config()` (mirrors existing config loaders)
- `configs/knowledge_graph.yaml` — settings: `neo4j_uri`, `neo4j_user`, `neo4j_password`, `neo4j_database`, `input_dir`, `batch_size`, `skip_existing`

**Tests:** `test_exceptions.py` (7), `test_models.py` (16), `test_config.py` (7)

**Dependencies:** None

### Subtask 2: Neo4jClient — Connection, Schema, MERGE Operations (~35 tests)

**Files:**
- `src/knowledge_graph/_client.py`:
  - `Neo4jClient(settings)` — lazy async driver init via `_ensure_driver()`
  - `async setup_schema()` — CREATE CONSTRAINT / INDEX statements from schema doc
  - `async merge_act(node: ActNode)` — idempotent MERGE
  - `async merge_section(node: SectionNode)` — idempotent MERGE
  - `async merge_section_version(node: SectionVersionNode, section: SectionNode)` — with HAS_VERSION
  - `async merge_judgment(node: JudgmentNode)` — idempotent MERGE
  - `async merge_amendment(node: AmendmentNode, section: SectionNode)` — with AMENDS/INSERTS/OMITS
  - `async merge_legal_concept(node: LegalConceptNode)` — idempotent MERGE
  - `async merge_court(node: CourtNode)` — idempotent MERGE
  - `async merge_judge(node: JudgeNode)` — idempotent MERGE
  - `async create_relationship(from_label, from_key, to_label, to_key, rel_type, properties=None)` — generic MERGE for relationships
  - `async execute_batch(operations: list)` — single transaction batch
  - `async close()` — clean shutdown

**Tests:** `test_client.py` (35) — mock `neo4j.AsyncGraphDatabase.driver()`, verify Cypher queries, test schema creation, batch execution, error handling, idempotent MERGE behavior

**Dependencies:** Subtask 1

### Subtask 3: Entity Extractors (~30 tests)

**Files:**
- `src/knowledge_graph/_extractors.py`:
  - `EntityExtractor` class:
    - `extract_from_statute_chunk(chunk: LegalChunk) -> ExtractedEntities` — Act + Section + SectionVersion + amendments
    - `extract_from_judgment_chunk(chunk: LegalChunk) -> ExtractedEntities` — Judgment + Court + judges
    - `extract_from_chunk(chunk: LegalChunk) -> ExtractedEntities` — router by document_type
    - `extract_legal_concepts(chunk: LegalChunk) -> list[LegalConceptNode]` — from definitions chunks
    - `_extract_year(act_number: str, date_enacted: date | None) -> int | None` — parse year from "Act No. 45 of 1860"
    - `_compute_text_hash(text: str) -> str` — SHA256
    - `_determine_act_status(metadata: StatuteMetadata) -> str`
    - `_determine_judgment_status(metadata: JudgmentMetadata) -> str`
  - `ExtractedEntities` model — holds lists of each node type extracted

**Tests:** `test_extractors.py` (30) — statute chunks, judgment chunks, definitions, amendments, edge cases (missing metadata, no section number, multiple judges), year extraction, text hashing

**Dependencies:** Subtask 1

### Subtask 4: Relationship Builder (~25 tests)

**Files:**
- `src/knowledge_graph/_relationships.py`:
  - `RelationshipBuilder` class:
    - `build_statute_relationships(chunk: LegalChunk, entities: ExtractedEntities) -> list[Relationship]` — CONTAINS, HAS_VERSION, amendment rels, REFERENCES (cross-section), DEFINES
    - `build_judgment_relationships(chunk: LegalChunk, entities: ExtractedEntities) -> list[Relationship]` — INTERPRETS, CITES_SECTION, CITES_CASE, OVERRULES, DISTINGUISHES, FOLLOWS, DECIDED_BY, FILED_IN
    - `build_from_chunk(chunk: LegalChunk, entities: ExtractedEntities) -> list[Relationship]` — router
    - `_parse_section_ref(ref: str) -> tuple[str, str] | None` — "Section 420 IPC" → ("Indian Penal Code", "420")
    - `_parse_case_citation(citation: str) -> str | None` — normalize citation
  - `Relationship` model — `from_label, from_key, to_label, to_key, rel_type, properties`

**Tests:** `test_relationships.py` (25) — statute cross-references, judgment citation building, amendment relationships, section→concept defines, edge cases (unparseable citations, missing acts)

**Dependencies:** Subtasks 1, 3

### Subtask 5: Query Builders (~25 tests)

**Files:**
- `src/knowledge_graph/_queries.py`:
  - `QueryBuilder(client: Neo4jClient)`:
    - All 8 query methods listed in the Critical Queries section above
    - Each returns typed results (Pydantic models or dicts)
    - All use parameterized Cypher (no string interpolation — injection safety)

**Tests:** `test_queries.py` (25) — mock Neo4j session responses, verify correct Cypher parameters, test point-in-time with edge dates, amendment cascading, citation traversal with court filter, hierarchy navigation, temporal status for repealed/amended sections, replacement lookup, node_exists for missing nodes

**Dependencies:** Subtask 2

### Subtask 6: Integrity Checker (~15 tests)

**Files:**
- `src/knowledge_graph/_integrity.py`:
  - `IntegrityChecker(client: Neo4jClient)`:
    - `async check_all() -> IntegrityReport` — runs all checks
    - `async check_section_versions()` — every Section has ≥1 SectionVersion
    - `async check_repealed_consistency()` — repealed Act → all sections `is_in_force=false`
    - `async check_overrule_hierarchy()` — OVERRULES only from equal/higher court
    - `async check_version_date_overlap()` — no overlapping effective_from/effective_until
  - `IntegrityReport` model — `passed: bool`, `checks: list[IntegrityCheck]`, each with `name, passed, violations: list[str]`

**Tests:** `test_integrity.py` (15) — one passing and one failing case per check, combined report

**Dependencies:** Subtasks 2, 5

### Subtask 7: Pipeline + CLI + Integration Tests (~30 tests)

**Files:**
- `src/knowledge_graph/pipeline.py` — `KnowledgeGraphPipeline`:
  - Constructor loads config, creates Neo4jClient, EntityExtractor, RelationshipBuilder
  - `async run(source_name=None, dry_run=False, skip_integrity=False) -> KGResult`
  - Discovery: scans `data/enriched/{source}/*.json`
  - Per-document: load chunks → extract entities → build relationships → batch upsert
  - Deduplication: track seen act names / citations to avoid redundant MERGE calls
  - Post-ingestion: run IntegrityChecker (unless `skip_integrity=True`)
  - Idempotency: MERGE-based, no skip_existing needed (MERGE is inherently idempotent)
- `src/knowledge_graph/run.py` — CLI: `--source`, `--dry-run`, `--skip-integrity`, `--log-level`, `--console-log`, `--config`
- `src/knowledge_graph/__main__.py` — `from src.knowledge_graph.run import main; main()`
- `src/knowledge_graph/__init__.py` — exports
- `tests/knowledge_graph/conftest.py` — shared fixtures (sample statute/judgment chunks, mock Neo4j driver, tmp dirs)

**Tests:** `test_pipeline.py` (15), `test_run.py` (7), `test_integration.py` (8) — end-to-end with mocked Neo4j: statute ingestion with amendments, judgment with citations, idempotency, error isolation, integrity check integration

**Dependencies:** All previous subtasks

## Test Summary

| Subtask | Tests |
|---------|-------|
| 1: Foundation | ~30 |
| 2: Neo4jClient | ~35 |
| 3: Entity Extractors | ~30 |
| 4: Relationship Builder | ~25 |
| 5: Query Builders | ~25 |
| 6: Integrity Checker | ~15 |
| 7: Pipeline + CLI + Integration | ~30 |
| **Total Phase 6** | **~190** |
| **Project total** | **~1,017** (827 + 190) |

## Acceptance Criteria

1. `python -m ruff check src/ tests/` — zero warnings
2. `python -m pytest tests/ -x` — all ~1,017 tests pass
3. `python -m src.knowledge_graph.run --dry-run` — exits 0, discovers enriched files
4. All 8 node types creatable from sample statute + judgment chunks
5. All 15 relationship types created with correct properties
6. Point-in-time query returns correct SectionVersion for past dates
7. Amendment cascade returns all affected sections for a given amending act
8. Citation traversal returns judgments interpreting a section, filterable by court
9. MERGE operations are idempotent: second run produces same graph state
10. Integrity checker catches: missing SectionVersion, inconsistent repeal status, invalid overrule hierarchy, overlapping version dates
11. All Neo4j calls mocked in unit tests — no real database required
12. Single document failure doesn't crash the pipeline

## Dependencies

**Required Python packages (all lazy-imported):**
- `neo4j` (async driver) — `pip install neo4j`

**No new non-optional dependencies.** Everything else (pydantic, structlog, pyyaml) is already installed.

**Infrastructure (for integration/production only):**
- Neo4j Community Edition 5.x (self-hosted or Docker: `docker compose up -d neo4j`)

## Critical Reference Files

- `docs/knowledge_graph_schema.md` — authoritative schema definition
- `docs/hallucination_mitigation.md` — how KG is used for citation/temporal verification
- `docs/pipeline_architecture.md` — Phase 6 position in pipeline, downstream consumers
- `src/embedding/_qdrant_indexer.py` — pattern for lazy external client + batch upserts
- `src/embedding/_config.py` — config loader pattern to mirror
- `src/embedding/_exceptions.py` — exception hierarchy pattern
- `src/chunking/_models.py` — LegalChunk model (input to extractors)
- `tests/embedding/` — test patterns for mocking external services

## Verification

After implementation:
```bash
python -m ruff check src/ tests/
python -m ruff format src/ tests/
python -m pytest tests/ -x -v
python -m pytest tests/knowledge_graph/ -v    # Phase 6 tests only
python -m src.knowledge_graph.run --dry-run   # Smoke test CLI
```
