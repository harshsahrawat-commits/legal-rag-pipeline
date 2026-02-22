Run the full ingestion pipeline on a single test document end-to-end.

This is the primary debugging and verification workflow. It processes ONE document through ALL implemented pipeline stages and reports results at each step.

1. Ask user for document path (or use `data/sample/indian_contract_act.pdf` as default)
2. Run each stage sequentially, printing results:

**Stage 1: Parse**
- Parse document with Docling
- Print: page count, extracted text length, tables found, OCR confidence (if scanned)

**Stage 2: Chunk**
- Run appropriate chunker based on document type
- Print: chunk count, avg/min/max token counts, chunk types distribution

**Stage 3: Enrich**
- Run Contextual Retrieval on first 3 chunks (to save API cost)
- Print: original text vs contextualized text for comparison
- Run QuIM-RAG on first 3 chunks
- Print: generated questions

**Stage 4: Embed & Index** (if Qdrant running)
- Embed chunks, index in Qdrant
- Print: vector count, index size

**Stage 5: Knowledge Graph** (if Neo4j running)
- Extract and insert nodes/relationships
- Print: nodes created, relationships created

**Stage 6: Test Retrieval**
- Ask user for a test query (or use default: "What are the elements of a valid contract?")
- Run full retrieval pipeline
- Print: top 5 chunks with scores, confidence score

Save full trace to `data/reports/ingest_test_{timestamp}.json`.
