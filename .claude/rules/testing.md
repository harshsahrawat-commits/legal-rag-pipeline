# Testing Conventions

- Test files mirror source structure: `src/parsing/docling_parser.py` → `tests/parsing/test_docling_parser.py`
- Use `pytest` with fixtures for shared setup (sample documents, mock API responses)
- Mark slow tests: `@pytest.mark.slow` (integration tests, API calls). Run fast tests by default.
- Use `pytest-asyncio` for async tests. Mark with `@pytest.mark.asyncio`.
- Sample test documents in `data/sample/` — small, representative files for each document type
- Mock external services (Anthropic API, Qdrant, Neo4j) in unit tests. Use real services only in integration tests.
- For chunking tests: validate chunk count, token bounds, metadata presence, and boundary correctness (no split sections)
- For parsing tests: compare against known-good extracted text in `data/expected/`
- Run `ruff check && ruff format && pytest tests/ -x` before every commit
