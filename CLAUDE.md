# Legal RAG Ingestion Pipeline — Indian Law

A production-grade ingestion pipeline for Indian legal documents (statutes, judgments, notifications, circulars). The system parses, chunks, enriches, embeds, indexes, and validates legal text for RAG-based retrieval that lawyers can cite in court.

## Stack

- **Language:** Python 3.11+ (dev environment: 3.14)
- **Parsing:** Docling (IBM, MIT License) + Granite-Docling VLM for scanned PDFs
- **Chunking:** Custom structure-aware chunkers per document type + semantic fallback
- **Embeddings:** Fine-tuned BGE-m3 (multilingual) with Late Chunking via Jina
- **Vector DB:** Qdrant (self-hosted, hybrid BM25+vector search)
- **Knowledge Graph:** Neo4j Community Edition
- **Queue/Orchestration:** Redis + Celery for async pipeline stages
- **LLM:** Multi-provider abstraction (Ollama primary, NVIDIA NIM fallback, Anthropic optional). Config: `configs/llm.yaml`
- **Evaluation:** RAGAS framework
- **Testing:** pytest + pytest-asyncio
- **Linting:** ruff (replaces flake8+isort+black)

## Project Structure

```
legal-rag-pipeline/
├── src/
│   ├── acquisition/       # Agentic source discovery, scrapers, change detection
│   ├── parsing/           # Document parsers (Docling wrappers, OCR pipeline)
│   ├── chunking/          # Structure-aware, semantic, RAPTOR chunkers
│   ├── enrichment/        # Contextual Retrieval, Late Chunking, QuIM-RAG
│   ├── embedding/         # Fine-tuned model inference, indexing
│   ├── knowledge_graph/   # Neo4j schema, ingestion, query builders
│   ├── hallucination/     # Citation verification, temporal checks, confidence scoring
│   ├── query/             # Query Intelligence Layer (cache, router, HyDE)
│   ├── retrieval/         # Hybrid search, reranking, fusion, FLARE
│   ├── evaluation/        # RAGAS pipeline, lawyer eval harness
│   └── utils/             # Shared helpers, logging, config
├── scripts/               # One-off scripts (bulk ingest, migration, benchmarks)
├── tests/                 # Mirrors src/ structure
├── data/                  # Sample documents for testing (NOT production data)
├── docs/                  # Detailed architecture docs (READ THESE — see below)
├── configs/               # YAML configs per environment
├── Indian_kanoon/         # Indian Kanoon API reference docs (endpoints, ToS, official client)
├── .claude/               # Claude Code commands, agents, rules
├── plans/                 # Implementation plans, tracked per phase
└── pyproject.toml         # Project config (dependencies, ruff, pytest)
```

## Commands

```bash
# Development
ruff check src/ tests/         # Lint
ruff format src/ tests/        # Format
pytest tests/ -x -v            # Run tests (stop on first failure)
pytest tests/ -k "unit" -v     # Unit tests only
pytest tests/ -k "integration" # Integration tests only

# Pipeline operations
python -m src.acquisition.run --source="Indian Kanoon" --mode=incremental
python -m src.parsing.run --source="Indian Kanoon" --dry-run
python -m src.chunking.run --source="Indian Kanoon" --dry-run
python -m src.enrichment.run --stage=contextual_retrieval --dry-run
python -m src.enrichment.run --stage=quim_rag --dry-run
python -m src.embedding.run --index
python -m src.evaluation --queries data/eval/test_queries.json --skip-ragas

# Infrastructure
docker compose up -d qdrant neo4j redis  # Start local services
```

## Code Conventions

- Type hints everywhere. Use `from __future__ import annotations`.
- Pydantic models for all data schemas (chunks, metadata, configs). Never raw dicts for structured data.
- Async where I/O bound (scraping, API calls, DB writes). Sync for CPU-bound (parsing, chunking).
- Every module exposes a clean interface via `__init__.py`. Internal details stay private (`_prefixed`).
- Errors: custom exception hierarchy rooted at `LegalRAGError`. Never bare `except:`.
- Logging: structured JSON logging via `structlog`. No print statements.
- Config: environment-specific YAML in `configs/`. Load via Pydantic Settings.
- Tests: every new module needs unit tests. Integration tests for cross-module flows.

## Architecture Rules

1. **Each pipeline stage is independent and idempotent.** You can re-run any stage without side effects. Stages communicate via files or message queue, never in-memory.
2. **Every chunk carries its full metadata.** A chunk must be self-describing — document type, source, section number, temporal status, etc. See `docs/metadata_schema.md`.
3. **Citation accuracy is non-negotiable.** Every section/case reference must be verified against the knowledge graph before reaching the user. See `docs/hallucination_mitigation.md`.
4. **Temporal correctness matters as much as factual correctness.** India replaced IPC/CrPC/Evidence Act in July 2024. The system must track which law is in force at any given date.
5. **Never commit API keys, credentials, or production data.** Use `.env` files (gitignored) and configs.

## Further Reading

**IMPORTANT:** Before starting any task, identify which docs below are relevant and read them first. Load the full context before making changes.

- `docs/pipeline_architecture.md` — Full 9-phase pipeline (Phase 0-9) with data flow diagrams
- `docs/query_intelligence.md` — Phase 0: semantic cache, query router, HyDE
- `docs/parsing_guide.md` — Docling configuration, fallback parsers, quality validation
- `docs/chunking_strategies.md` — 6 chunking strategies + tiered routing + RAPTOR + QuIM-RAG
- `docs/enrichment_guide.md` — Late Chunking + Contextual Retrieval + prompt caching patterns
- `docs/knowledge_graph_schema.md` — Neo4j node/relationship types, Cypher patterns
- `docs/metadata_schema.md` — Complete Pydantic models for chunk metadata (incl. parent-doc fields)
- `docs/hallucination_mitigation.md` — 5-layer defense: citation, temporal, confidence, GenGround, Finetune-RAG
- `docs/embedding_fine_tuning.md` — QLoRA + NEFTune + DPO fine-tuning for Indian legal embeddings
- `docs/indian_legal_structure.md` — How Indian statutes/judgments are structured (for non-lawyers)
- `docs/evaluation_framework.md` — RAGAS + latency + routing metrics, monitoring cadence

## Git Workflow

- Branch from `main` for each pipeline phase: `phase-1/parsing`, `phase-2/chunking`, etc.
- Commit frequently with descriptive messages. Prefix: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
- Run `ruff check && ruff format && pytest tests/ -x` before every commit.
- Tag milestones: `v0.1-parsing`, `v0.2-chunking`, etc.

## Working With This Codebase

- Start in **plan mode** for any new phase or feature. Save plan to `plans/`.
- For large features, break into subtasks and use agent teams for parallel work.
- After completing a task, update the relevant doc in `docs/` with any new decisions or gotchas.
- If you discover a pattern that should be universal, add it to this file. If it's module-specific, add it to the relevant `docs/` file or subdirectory CLAUDE.md.
