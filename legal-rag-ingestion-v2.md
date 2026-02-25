# Legal RAG Ingestion Pipeline v2: Production-Grade Architecture for Indian Law

## Document Purpose & Scope

This document is the complete technical blueprint for building one of the best legal RAG ingestion pipelines currently possible. It synthesizes research from 30+ papers, benchmarks from Applied AI's PDFBench (800+ documents), Harvey AI's enterprise architecture, the Stanford hallucination study (2025), and cutting-edge techniques including Late Chunking, QuIM-RAG, Finetune-RAG, and Structure-Aware Temporal Graph RAG.

The target: a system that a lawyer can trust with their career. Not a demo. Not a prototype. A system where every citation is verifiable, every law is temporally correct, and every response is grounded in actual Indian legal text.

---

## PART I: THE COMPETITIVE LANDSCAPE — WHAT "BEST" LOOKS LIKE

Before building, you need to understand what you're competing against and where the bar actually sits.

### Harvey AI — The $5B+ Benchmark

Harvey serves 97% of Am Law 100 firms across 45 countries. Key architectural decisions relevant to your system:

- **Three RAG data sources:** User-uploaded files, long-term vault projects, and third-party legal databases. Each has different privacy, latency, and freshness requirements.
- **LanceDB Enterprise** as primary vector DB (chosen over Postgres/pgvector after systematic evaluation) for latency, accuracy, ingestion throughput, and data privacy.
- **Hybrid sparse+dense retrieval** — they explicitly acknowledge that purely dense embeddings miss legal identifiers, case citations, and named entities.
- **Custom-trained models with OpenAI** — they injected 10 billion tokens of case law into a custom model. 97% of lawyers preferred this over GPT-4 for case law research.
- **"Data Factory"** — an autonomous multi-agent pipeline that expanded from 20 to 400+ data sources since August 2025, using Sourcing Agents, Legal Review Agents, and Deployment Agents.
- **Continuous one-way sync** — documents auto-update when sources change. No manual re-uploads.
- **Domain expert collaboration** — PwC tax professionals helped build their Tax AI Assistant, which achieved 91% preference over ChatGPT.

**Key lesson from Harvey:** Their founder Winston Weinberg said: "If you just do retrieval, you can answer very simple questions about areas of law that you aren't really an expert in, but that's actually not that useful for most attorneys. With case law research, you're finding ammo for your argument, and that's much more difficult to do." RAG alone isn't enough — you need deep domain reasoning.

### The Stanford Hallucination Study — The Hard Truth

A preregistered 2025 study from Stanford (Journal of Empirical Legal Studies) evaluated commercial legal RAG tools and found:

- **Lexis+ AI (best performer):** 65% accuracy, hallucinating 17-33% of the time
- **Westlaw AI-Assisted Research:** 42% accuracy, hallucinating nearly twice as often
- **GPT-4 (no RAG):** Hallucinated at least 49% on basic case summary tasks
- **Vincent AI (RAG-powered, academic study):** Showed statistically significant quality gains on 4 of 6 legal tasks vs. non-RAG tools

Provider claims of "hallucination-free" legal AI are empirically false. This means your system needs an explicit hallucination mitigation layer — not as an afterthought, but as a core architectural component.

### The Actual Hierarchy of Legal RAG Quality

Based on all research reviewed:

```
Tier 1: Harvey, custom-trained models + enterprise RAG + multi-agent pipelines
Tier 2: Lexis+ AI, Thomson Reuters — large databases, but hallucination rates of 17-33%
Tier 3: Well-engineered RAG with knowledge graphs, fine-tuned embeddings, hybrid search
Tier 4: Basic RAG with off-the-shelf embeddings and naive chunking
Tier 5: Raw LLM with no retrieval (49%+ hallucination on legal tasks)
```

Your target: **Tier 3 at launch, with a roadmap to Tier 2.** Tier 1 requires $500M+ in funding and custom model training partnerships. Tier 3, done right for the Indian market, is still a massive opportunity since nothing comparable exists.

---

## PART II: THE EIGHT-PHASE PIPELINE

### Phase 1: Agentic Document Acquisition

Static web scrapers are a maintenance nightmare. Indian government websites change URL structures, go down periodically, and publish in inconsistent formats. Build agentic acquisition from day one.

#### Source Map for Indian Legal Data

| Source | Content | Format | Volume Estimate | Update Frequency |
|--------|---------|--------|-----------------|------------------|
| Indian Kanoon | Case law, statutes | HTML | 70M+ pages | Daily |
| India Code (indiacode.nic.in) | Central Acts (800+) | HTML/PDF | ~50,000 sections | On amendment |
| State Legislature websites (28 states + 8 UTs) | State Acts | PDF (often scanned) | Varies widely | Quarterly |
| Gazette of India (egazette.gov.in) | Notifications, rules, orders | PDF | ~5,000/year | Weekly |
| Supreme Court (sci.gov.in) | Judgments | PDF | ~30,000/year | Daily |
| High Court websites (25 HCs) | Judgments | PDF/HTML | ~500,000/year total | Daily |
| Tribunal websites (NCLT, NCLAT, ITAT, SAT, NGT, etc.) | Orders | PDF | ~100,000/year | Daily |
| Law Commission of India | Reports, recommendations | PDF | ~280 total | Sporadic |
| Bar Council, RBI, SEBI, TRAI circulars | Regulatory guidance | PDF/HTML | Thousands per regulator | Weekly |

#### Agentic Acquisition Architecture

Inspired by Harvey's Data Factory, build three specialized agents:

**Agent 1: Source Discovery Agent**
- Maintains a registry of all known Indian legal data sources
- Periodically probes each source URL for structural changes
- Discovers new sources (e.g., when a new tribunal is constituted)
- Outputs: Updated source manifest with URL patterns, access methods, and document type classifications

**Agent 2: Change Detection Agent**
- Runs on schedule (daily for courts, weekly for gazette, monthly for legislation portals)
- Compares current site state against last known state
- Detects new documents, modified documents, and removed/replaced documents
- For Indian Kanoon: Monitors the "recently added" feed
- For Supreme Court: Checks the daily cause list and judgment archive
- Outputs: Queue of documents requiring ingestion/re-ingestion

**Agent 3: Legal Review Agent**
- Pre-analyzes each new document before ingestion
- Classifies document type (statute/judgment/notification/circular/order)
- Extracts preliminary metadata (court, date, parties, Act references)
- Flags potential issues (scanned PDF needing OCR, regional language, corrupted file)
- Outputs: Classified, prioritized ingestion queue with metadata

**Implementation notes:**
- Use a message queue (Redis/RabbitMQ) between agents
- Store source manifest in a version-controlled config (so you can track when source structures changed)
- Log every acquisition action for audit trail
- Build retry logic with exponential backoff (Indian government sites are unreliable)
- Respect robots.txt and rate limits — you don't want your IPs blocked

**Why this matters:** Harvey learned the hard way that static ingestion breaks at scale. They rebuilt their entire file ingestion architecture in early 2025 specifically to solve this problem. By starting agentic, you skip that painful migration later.

---

### Phase 2: Document Parsing

#### Primary Parser: Docling (IBM, MIT License)

**Why Docling wins for your use case:**
- Open-source, MIT license — no licensing costs, no vendor lock-in
- Runs locally — critical for legal data privacy (privileged documents never leave your servers)
- Advanced PDF understanding: page layout, reading order, table structure, code, formulas
- Unified DoclingDocument format preserves document hierarchy
- Native integrations with LangChain, LlamaIndex, Haystack
- Custom PDF parser (docling-parse) built on qpdf — handles edge cases better than pymupdf/pypdfium2
- Two pipeline modes: Classic (multi-model ensemble) and VLM (Granite-Docling, 258M params)

**Pipeline configuration for Indian legal documents:**

```python
# Pseudocode — actual Docling implementation
from docling.document_converter import DocumentConverter
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

converter = DocumentConverter()

# For digitally-born PDFs (Supreme Court, recent legislation)
# Use standard pipeline — fast, accurate for clean documents
result = converter.convert("judgment.pdf")
doc = result.document

# For scanned PDFs (older state legislation, some HC judgments)
# Enable OCR pipeline with Granite-Docling VLM
# This handles mixed Hindi/English text
result = converter.convert("old_state_act.pdf", ocr=True)

# Export to structured format preserving hierarchy
markdown = doc.export_to_markdown()  # For text
json_doc = doc.export_to_json()      # For programmatic access with metadata
```

#### Fallback Parsers (For Specific Edge Cases)

| Scenario | Parser | Reasoning |
|----------|--------|-----------|
| Batch processing 100K+ files fast | LlamaParse ($0.003/page) | Consistent ~6s regardless of page count |
| Complex tables in financial regulations | Reducto API | Multi-pass OCR+VLM, best table accuracy |
| Very old scanned documents with poor quality | Tesseract OCR + manual review queue | No AI parser handles degraded scans reliably |

#### Parsing Quality Validation

After parsing every document, run automated quality checks:

1. **Text completeness:** Compare extracted text length against expected length (based on page count)
2. **Structure preservation:** Verify section numbers are sequential and complete
3. **Table integrity:** Check that table rows/columns are properly aligned
4. **OCR confidence:** For scanned documents, Tesseract provides per-character confidence — flag documents below 85%
5. **Cross-reference with Indian Kanoon:** For judgments that exist on Indian Kanoon, compare extracted text against their HTML version as ground truth

**Benchmark targets:**
- Text extraction accuracy: >95% on digitally-born PDFs
- Structure preservation (section hierarchy): >90%
- Table extraction accuracy: >85%
- OCR accuracy on scanned Hindi/English: >80% (flag below this for manual review)

---

### Phase 3: Structure-Aware Chunking

This is the highest-leverage decision in the entire pipeline. Research consistently shows chunking quality directly determines retrieval quality, with up to 9% recall gap between best and worst strategies.

#### The Core Principle: Legal Documents Have Natural Semantic Boundaries — Use Them

Unlike generic text, legal documents have predefined, hierarchical structures that perfectly align with how lawyers think and query:

**Indian Statutes:**
```
Act
├── Part (e.g., Part III - Fundamental Rights)
│   ├── Chapter (e.g., Chapter III - Right Against Exploitation)
│   │   ├── Section (e.g., Section 23 - Prohibition of traffic in human beings)
│   │   │   ├── Sub-section (1), (2), (3)
│   │   │   ├── Clause (a), (b), (c)
│   │   │   ├── Proviso ("Provided that...")
│   │   │   └── Explanation
│   │   └── Section 24...
│   └── Chapter IV...
└── Schedule(s)
```

**Indian Court Judgments:**
```
Judgment
├── Header (Court name, case number, date, bench composition)
├── Parties (Petitioner vs. Respondent)
├── Facts of the case
├── Issues framed by the court
├── Arguments by Petitioner's counsel
├── Arguments by Respondent's counsel  
├── Analysis & Reasoning (per issue)
│   ├── Issue 1: Reasoning + cited precedents + statutory provisions
│   ├── Issue 2: Reasoning + cited precedents + statutory provisions
│   └── ...
├── Holding / Ratio decidendi
├── Obiter dicta
├── Final Order / Disposition
└── Costs (if any)
```

#### Strategy 1: Structure-Boundary Chunking (Primary — For Statutes)

Chunk at section boundaries, keeping sub-sections, provisos, and explanations together with their parent section.

```
Chunk boundary rules for Indian statutes:
1. Each Section = one chunk (including all sub-sections, clauses, provisos, explanations)
2. If Section exceeds 1500 tokens, split at sub-section boundaries
3. If a single sub-section exceeds 1500 tokens (rare), use semantic chunking as fallback
4. Definitions sections: Each defined term = one chunk
5. Schedules: Each schedule entry = one chunk
6. Always include: Section number + Act name + Chapter/Part in every chunk
```

**Why 1500 tokens as the upper bound:** Legal sections in Indian statutes average 200-800 tokens. Some complex sections (like Section 138 NI Act with its provisos) can reach 1200+. The 1500 ceiling accommodates these while keeping chunks focused enough for precise retrieval. Research shows analytical queries benefit from 1024+ token chunks.

#### Strategy 2: Structural Chunking (Primary — For Judgments)

```
Chunk boundary rules for Indian judgments:
1. Header/metadata = one chunk (compact, high-metadata)
2. Facts = one or more chunks (split at paragraph boundaries if >1500 tokens)
3. Each issue + its reasoning = one chunk
4. Final holding/order = one chunk
5. Each dissenting opinion = separate chunk(s)
6. Always include: Case citation + court + date + bench in every chunk
```

#### Strategy 3: Semantic Chunking (Fallback — For Unstructured Documents)

For documents without clean structural markers (some circulars, notifications, law commission reports), use the **Max-Min semantic chunking algorithm** (Springer, June 2025):

- Split into sentences → embed each sentence → compare similarity between consecutive sentences
- Split where similarity drops below a threshold (using percentile, standard deviation, or IQR method)
- Achieved AMI scores of 0.85-0.90, significantly outperforming fixed-size chunking

#### Strategy 4: RAPTOR — Hierarchical Summary Trees (Enhancement Layer)

Build RAPTOR trees on top of the base chunks for every Act and major judgment set. This enables multi-hop reasoning across document sections.

```
RAPTOR Tree for the Indian Contract Act, 1872:

Level 0 (Root): "The Indian Contract Act, 1872 is a comprehensive statute governing
                 the formation, performance, and breach of contracts in India..."

Level 1 (Chapter summaries):
├── "Chapter I (Sections 1-2): Preliminary, defines the scope and application..."
├── "Chapter II (Sections 3-9): Communication, acceptance, and revocation of proposals..."
├── "Chapter III (Sections 10-30): Defines what constitutes a valid contract..."
└── ...

Level 2 (Section-level): The actual section chunks from Strategy 1
```

**How this helps:** When a lawyer asks "Under what circumstances can a contract be deemed void?", the system can first identify from the Level 1 summary that Chapter II (Void Agreements, Sections 24-30) is relevant, then retrieve the specific section-level chunks. Without RAPTOR, the system might miss sections that use different terminology but address the same legal concept.

#### Strategy 5: QuIM-RAG — Inverted Question Matching (Enhancement Layer)

For every chunk, pre-generate 3-5 questions that the chunk could answer. Embed these questions and use them as an additional retrieval pathway.

**Why this is critical for legal text:**

Consider this chunk from the Rent Control Act:
> "No order for the recovery of possession of any premises shall be made on the ground specified in clause (e) of the proviso to sub-section (1) unless the Court is satisfied that the claim of the landlord is bona fide."

A lawyer's query: "Can a landlord evict a tenant for personal use?"

The semantic gap between the query and the chunk is large — the chunk never uses the words "evict," "tenant," or "personal use." But if you pre-generate questions like:
- "Under what conditions can a landlord recover possession?"
- "What must a court be satisfied about before granting eviction?"
- "What is the bona fide requirement for landlord possession claims?"

...the match becomes much tighter. Research shows this reduces information dilution and hallucination compared to passage-level retrieval alone.

**Implementation:**
```python
# Pseudocode for QuIM-RAG question generation
for chunk in all_chunks:
    prompt = f"""Given this legal text, generate 3-5 questions that a lawyer 
    might ask which this text could answer. Focus on practical legal queries, 
    not academic ones.
    
    Legal text: {chunk.text}
    Context: This is from {chunk.act_name}, {chunk.section_number}
    
    Generate questions:"""
    
    questions = llm.generate(prompt)
    
    # Embed questions alongside the chunk
    for q in questions:
        question_embedding = embed(q)
        store_in_vector_db(
            embedding=question_embedding,
            metadata={"type": "question", "source_chunk_id": chunk.id},
            text=q
        )
```

**Cost estimate:** At ~50 tokens per question generation × 5 questions × number of chunks, using Claude Haiku or a similar cheap model. For 100,000 chunks, this costs roughly $20-40.

---

### Phase 4: Dual Context Enrichment (Late Chunking + Contextual Retrieval)

This is where the pipeline diverges from standard approaches. Most systems use either Anthropic's Contextual Retrieval OR Late Chunking. **You should use both**, because they solve the same problem from complementary angles.

#### Late Chunking — For Superior Vector Embeddings

**What it is (arXiv 2409.04701, Jina AI, July 2025 revision):**

Instead of the traditional approach (chunk first → embed each chunk independently), Late Chunking reverses the order:

```
Traditional: Document → Split into chunks → Embed each chunk separately
Late Chunking: Document → Embed entire document (all tokens) → THEN split token embeddings into chunks → Mean pool each chunk
```

**Why this matters for legal text:**

The "lost context" problem is devastating for legal documents. When a statute says "Whoever commits an offence under this section shall be punished...", traditional chunking produces an embedding that has no idea what "this section" refers to. With Late Chunking, the transformer has already processed the entire document, so the token embeddings for "this section" contain the information that it refers to Section 420 IPC (cheating).

**Results from the paper:**
- Late Chunking always outperforms naive chunking across all BeIR datasets tested
- The improvement scales with document length — longer documents benefit more
- Requires NO additional training — works with any long-context embedding model
- Similarity score improvements were "mind-blowing" for chunks containing anaphoric references

**How to implement:**
```python
# Pseudocode for Late Chunking
from jina_embeddings import JinaEmbeddingModel  # or any long-context model

model = JinaEmbeddingModel("jina-embeddings-v3")  # supports 8192 tokens

def late_chunk_embed(document_text, chunk_boundaries):
    # Step 1: Get ALL token embeddings for entire document
    token_embeddings = model.encode_tokens(document_text)  # shape: [num_tokens, dim]
    
    # Step 2: Split token embeddings at chunk boundaries
    chunk_embeddings = []
    for start, end in chunk_boundaries:
        chunk_tokens = token_embeddings[start:end]
        # Step 3: Mean pool each chunk's tokens
        chunk_embedding = chunk_tokens.mean(dim=0)
        chunk_embeddings.append(chunk_embedding)
    
    return chunk_embeddings
```

**Cost:** Near-zero marginal cost. You're already embedding documents — Late Chunking just changes the order of operations. The only requirement is a long-context embedding model (jina-embeddings-v3 supports 8192 tokens, which covers ~10 pages).

#### Anthropic's Contextual Retrieval — For Superior BM25 Matching

Late Chunking improves vector embeddings, but it doesn't help BM25 keyword search. For that, you still need Contextual Retrieval.

**What it does:** For each chunk, use an LLM to prepend a brief context that situates it within the document.

```
Before:
"Whoever commits an offence under this section shall be punished with imprisonment 
for a term which may extend to seven years."

After Contextual Retrieval:
"[Context: Section 420 of the Indian Penal Code, 1860, Chapter XVII - Offences Against 
Property. This section addresses punishment for cheating and dishonestly inducing delivery 
of property.] Whoever commits an offence under this section shall be punished with 
imprisonment for a term which may extend to seven years."
```

Now when a BM25 search for "cheating punishment IPC" runs, this chunk will match — even though the original text never contained the word "cheating."

**Performance (Anthropic's testing):**
| Technique | Retrieval Failure Rate Reduction |
|-----------|--------------------------------|
| Contextual Embeddings alone | 35% |
| Contextual Embeddings + Contextual BM25 | 49% |
| + Reranking | **67%** |

**Cost optimization with prompt caching:**
- Cache the full document text in the system prompt
- For each chunk, only the chunk-specific query varies
- With Claude Haiku + prompt caching: ~$1.02 per million document tokens
- For 100K chunks across all your documents: ~$50-100 total

#### Why Use Both Together

| Aspect | Late Chunking | Contextual Retrieval |
|--------|--------------|---------------------|
| Improves vector search | Yes (directly) | Yes (indirectly, via enriched text) |
| Improves BM25 search | No | Yes (adds keywords) |
| Cost per chunk | ~$0 (compute only) | ~$0.001 (LLM call) |
| Requires LLM | No | Yes |
| Works with any embedding model | Needs long-context model | Works with any model |
| Also helps generation quality | No | Yes (context passed to LLM) |

**Combined approach:**
1. Use Late Chunking for generating vector embeddings (superior semantic matching)
2. Use Contextual Retrieval for enriching the BM25 index (superior keyword matching)
3. Use the contextualized text when passing retrieved chunks to the LLM for generation (better grounding)

This gives you the 67% retrieval failure reduction from Contextual Retrieval PLUS the additional improvements from Late Chunking on the vector side. No other pipeline I've found in research combines both.

---

### Phase 5: Metadata Extraction & Knowledge Graph

#### Essential Metadata Schema

Every chunk must carry rich metadata. This isn't optional — metadata enables filtering, temporal queries, and citation verification.

```json
{
  "chunk_id": "uuid-v4",
  "document_id": "uuid-v4",
  "document_type": "statute | judgment | notification | circular | order | report",
  
  "source": {
    "url": "https://indiankanoon.org/doc/...",
    "scraped_at": "2025-02-22T00:00:00Z",
    "last_verified": "2025-02-22T00:00:00Z"
  },
  
  "legal_metadata": {
    "act_name": "Indian Penal Code, 1860",
    "act_number": "Act No. 45 of 1860",
    "section_number": "420",
    "chapter": "XVII",
    "part": null,
    "schedule": null,
    
    "jurisdiction": "Central",
    "applicable_states": ["All India"],
    
    "date_enacted": "1860-10-06",
    "date_effective": "1862-01-01",
    "date_repealed": "2024-07-01",
    "repealed_by": "Bharatiya Nyaya Sanhita, 2023 (Act No. 45 of 2023)",
    "replaced_by_section": "Section 318 BNS",
    
    "is_currently_in_force": false,
    "temporal_status": "repealed",
    "last_amended_by": null,
    "amendment_history": [
      {"amending_act": "...", "date": "...", "nature": "substitution | insertion | omission"}
    ]
  },
  
  "judgment_metadata": {
    "case_citation": "AIR 2023 SC 1234",
    "alternative_citations": ["(2023) 5 SCC 678", "2023 SCC OnLine SC 890"],
    "court": "Supreme Court of India",
    "court_hierarchy_level": 1,
    "bench_type": "Division Bench",
    "bench_strength": 2,
    "judge_names": ["Justice A.B. Sharma", "Justice C.D. Verma"],
    "date_decided": "2023-05-15",
    "case_type": "Criminal Appeal",
    "parties": {
      "petitioner": "State of Maharashtra",
      "respondent": "Rajesh Kumar"
    },
    "case_status": "decided",
    "overruled": false,
    "overruled_by": null,
    "distinguished_in": ["AIR 2024 SC 567"],
    "followed_in": ["AIR 2023 Bom 890", "AIR 2024 Del 123"]
  },
  
  "content_metadata": {
    "sections_cited": ["Section 302 IPC", "Section 34 IPC", "Section 149 IPC"],
    "acts_cited": ["Indian Penal Code, 1860", "Code of Criminal Procedure, 1973"],
    "cases_cited": ["K.M. Nanavati v. State of Maharashtra, AIR 1962 SC 605"],
    "legal_concepts": ["murder", "common intention", "unlawful assembly"],
    "chunk_type": "reasoning | facts | holding | statutory_text | proviso | definition",
    "language": "en",
    "has_hindi_text": false
  },
  
  "ingestion_metadata": {
    "ingested_at": "2025-02-22T00:00:00Z",
    "parser_used": "docling_v2",
    "ocr_confidence": null,
    "parsing_quality_score": 0.95,
    "chunk_strategy": "structure_boundary",
    "contextualized": true,
    "late_chunked": true,
    "quim_questions_generated": 5,
    "raptor_summary_available": true
  }
}
```

#### Knowledge Graph Schema (Neo4j)

The knowledge graph is what enables temporal queries, citation traversal, and amendment tracking that pure vector search cannot do.

```cypher
// Node types
(:Act {name, number, year, date_enacted, date_effective, date_repealed, 
       jurisdiction, status: "in_force"|"repealed"|"partially_repealed"})

(:Section {number, text_hash, parent_act, chapter, part,
           date_inserted, date_last_amended, is_currently_in_force})

(:SectionVersion {version_id, text, effective_from, effective_until,
                  amending_act, nature: "original"|"substituted"|"inserted"})

(:Judgment {citation, court, date_decided, bench_type, bench_strength,
            case_type, status: "good_law"|"overruled"|"distinguished"})

(:Amendment {amending_act, date, gazette_notification, nature})

(:LegalConcept {name, definition_source, category})

(:Court {name, hierarchy_level, state, jurisdiction_type})

(:Judge {name, courts_served})

// Relationships
(:Act)-[:CONTAINS]->(:Section)
(:Section)-[:HAS_VERSION]->(:SectionVersion)
(:Amendment)-[:AMENDS]->(:Section)
(:Amendment)-[:INSERTS]->(:Section)
(:Amendment)-[:OMITS]->(:Section)
(:Act)-[:REPEALS]->(:Act)
(:Act)-[:REPLACES]->(:Act)
(:Judgment)-[:INTERPRETS]->(:Section)
(:Judgment)-[:CITES_SECTION]->(:Section)
(:Judgment)-[:CITES_CASE]->(:Judgment)
(:Judgment)-[:OVERRULES]->(:Judgment)
(:Judgment)-[:DISTINGUISHES]->(:Judgment)
(:Judgment)-[:FOLLOWS]->(:Judgment)
(:Judgment)-[:DECIDED_BY]->(:Judge)
(:Judgment)-[:FILED_IN]->(:Court)
(:Section)-[:REFERENCES]->(:Section)
(:Section)-[:DEFINES]->(:LegalConcept)
```

**Critical capability this enables — Point-in-time retrieval:**

```cypher
// "What was Section 420 IPC as on January 1, 2020?"
MATCH (s:Section {number: "420"})-[:HAS_VERSION]->(v:SectionVersion)
WHERE v.effective_from <= date("2020-01-01")
  AND (v.effective_until IS NULL OR v.effective_until > date("2020-01-01"))
RETURN v.text

// "Which sections of CrPC were replaced by BNSS?"
MATCH (old:Act {name: "Code of Criminal Procedure, 1973"})
      -[:CONTAINS]->(s:Section),
      (new:Act {name: "Bharatiya Nagarik Suraksha Sanhita, 2023"})
      -[:REPLACES]->(old)
RETURN s.number, s.replaced_by_section

// "Find all Supreme Court judgments that interpret Section 498A IPC"
MATCH (j:Judgment)-[:INTERPRETS]->(s:Section {number: "498A"})
WHERE j.court = "Supreme Court of India"
RETURN j ORDER BY j.date_decided DESC
```

This is directly inspired by the SAT-Graph RAG paper (arXiv 2505.00039) which demonstrated that structure-aware temporal graphs enable deterministic, verifiable legal retrieval — something flat vector search fundamentally cannot provide.

---

### Phase 6: Embedding, Indexing & Retrieval

#### Fine-Tuned Legal Embedding Model — Not Optional

Generic embedding models break on Indian legal text. A practitioner who fine-tuned BGE-base-en-v1.5 on SEBI regulatory text achieved **16% performance gains with 12x storage reduction** using Matryoshka Representation Learning.

**Why generic models fail on Indian law:**
- Regulatory terminology: "debenture trustee," "dematerialisation," "cognizable offence"
- Cross-referential structure: "Section 23A refers to Section 12B which is subject to the proviso under Section 8(2)(c)"
- Mixed English + Hindi + legal Latin: "mens rea," "suo motu," "locus standi" mixed with English text
- Indian-specific legal concepts with no direct Western equivalent: "anticipatory bail," "lok adalat," "First Information Report"

**Fine-tuning plan:**

```
Phase 1: Create training dataset
├── Collect 50K+ query-document pairs from Indian legal text
├── Sources: Indian Kanoon search logs (if accessible), 
│   manually curated Q&A pairs, LLM-generated pairs from statutes
├── Include hard negatives (similar but wrong sections)
└── Include cross-referential pairs (question about Section X, 
    answer in Section Y that references X)

Phase 2: Fine-tune base model
├── Base model: BAAI/bge-base-en-v1.5 or BAAI/bge-m3 (multilingual)
├── Training: MatryoshkaLoss for multi-dimensional efficiency
├── Hyperparameters (from SEBI fine-tuning success):
│   ├── Epochs: 4
│   ├── Batch size: 32 per device, gradient accumulation 16
│   ├── Learning rate: 2e-5 with cosine scheduler
│   ├── Precision: bf16 + tf32
│   └── Attention: flash_attention
└── Evaluate on held-out Indian legal Q&A pairs

Phase 3: Deploy with Late Chunking support
├── Model must support long context (ideally 8192 tokens)
├── If bge-base doesn't support long context natively,
│   consider jina-embeddings-v3 as base instead
└── Deploy on GPU server for inference
```

**Hardware requirement:** Single GPU with 10-12GB VRAM minimum for fine-tuning. For inference, CPU is fine for moderate query volumes.

#### Hybrid Search Architecture

```
User Query
    │
    ├──→ [Dense Retrieval] Late-chunked embeddings via fine-tuned model
    │    → Top 100 by cosine similarity
    │
    ├──→ [Sparse Retrieval] BM25 on contextualized chunks
    │    → Top 100 by BM25 score
    │
    ├──→ [Question Matching] QuIM-RAG pre-generated questions
    │    → Top 50 by cosine similarity to query
    │
    ├──→ [Graph Retrieval] Knowledge graph traversal
    │    → Sections cited by/citing retrieved chunks
    │    → Amendment history of relevant sections
    │    → Related judgments from citation graph
    │
    └──→ [Reciprocal Rank Fusion]
         Combine all results → Deduplicate → Top 150
              │
              └──→ [Cross-Encoder Reranking]
                   Rerank top 150 → Select top 20
                        │
                        └──→ Pass to LLM for generation
```

#### Vector Database: Qdrant (Recommended)

| Feature | Qdrant | Weaviate | Pinecone | pgvector |
|---------|--------|----------|----------|----------|
| Hybrid search (vector + BM25) | Native | Native | Limited | No BM25 |
| Self-hosted | Yes | Yes | No | Yes |
| Performance at scale | Excellent (Rust) | Good | Excellent | Moderate |
| Filtering on metadata | Very fast | Good | Good | SQL (flexible) |
| Cost | Free (self-hosted) | Free (self-hosted) | $70+/month | Free (self-hosted) |
| Multi-tenancy | Native support | Yes | Yes | Via schemas |

Qdrant is the best fit: self-hosted (data privacy), native hybrid search, fast metadata filtering (critical for jurisdiction/date filters), and free.

#### Reranking

Use a cross-encoder model to rerank the top 150 fused results down to 20:

- **Best option:** BGE-reranker-v2-m3 (open-source, multilingual — handles Hindi)
- **Alternative:** Cohere Rerank (commercial, higher quality, but adds API dependency)
- **Why 150→20:** Anthropic's research found 20 chunks was optimal — fewer misses important context, more overwhelms the LLM

---

### Phase 7: Hallucination Mitigation — The Critical Layer Most Pipelines Skip

The Stanford study proved that even the best commercial legal RAG tools hallucinate 17-33% of the time. This layer is what separates a product lawyers trust from a liability.

#### Layer 7a: Citation Verification (Pre-Response)

Before returning any response to the user, verify every citation against the knowledge graph:

```python
# Pseudocode for citation verification
def verify_citations(llm_response, knowledge_graph):
    # Extract all citations from the response
    citations = extract_citations(llm_response)  
    # e.g., ["Section 420 IPC", "AIR 2023 SC 1234", "Article 21"]
    
    verified = []
    flagged = []
    
    for citation in citations:
        # Check if citation exists in knowledge graph
        exists = knowledge_graph.verify_exists(citation)
        
        if not exists:
            flagged.append({"citation": citation, "issue": "not_found"})
            continue
        
        # Check if the proposition attributed to the citation is accurate
        source_text = knowledge_graph.get_text(citation)
        attribution_correct = llm_check_attribution(
            claim=extract_claim_for_citation(llm_response, citation),
            source=source_text
        )
        
        if not attribution_correct:
            flagged.append({"citation": citation, "issue": "misattribution"})
        else:
            verified.append(citation)
    
    return verified, flagged
```

**If any citations are flagged:**
- Remove the flagged citation from the response
- Add a disclaimer: "Note: [X] citations could not be verified and have been removed"
- Log the hallucination for model improvement

#### Layer 7b: Temporal Consistency Check

Verify that all referenced laws are currently in force (unless the user asked about historical law):

```python
def check_temporal_consistency(response, query_date=None):
    # Default to current date if user didn't specify a historical context
    reference_date = query_date or datetime.now()
    
    for section_ref in extract_section_references(response):
        status = knowledge_graph.get_temporal_status(section_ref, reference_date)
        
        if status == "repealed":
            # Flag: "Section 420 IPC has been repealed by BNS Section 318 
            # effective July 1, 2024. The current applicable provision is..."
            add_warning(response, section_ref, status)
        
        elif status == "amended":
            # Flag: "Note: Section X was amended by [Act] on [date]. 
            # The version shown may not reflect the latest amendment."
            add_warning(response, section_ref, status)
```

**This is a killer feature.** India recently replaced three major criminal codes (IPC→BNS, CrPC→BNSS, Indian Evidence Act→BSA) effective July 1, 2024. A system that doesn't track this will confidently cite repealed law — which is worse than no answer at all.

#### Layer 7c: Confidence Scoring

For every response, calculate and display a confidence score:

```python
def calculate_confidence(query, retrieved_chunks, response):
    factors = {
        "retrieval_relevance": avg([chunk.similarity_score for chunk in retrieved_chunks]),
        "source_authority": max_court_hierarchy(retrieved_chunks),  # SC > HC > District
        "source_recency": recency_score(retrieved_chunks),
        "citation_verification_rate": len(verified) / len(total_citations),
        "chunk_agreement": measure_agreement_between_top_chunks(retrieved_chunks),
        "query_specificity": measure_query_specificity(query)
    }
    
    # Weighted combination
    confidence = (
        0.25 * factors["retrieval_relevance"] +
        0.20 * factors["citation_verification_rate"] +
        0.20 * factors["source_authority"] +
        0.15 * factors["chunk_agreement"] +
        0.10 * factors["source_recency"] +
        0.10 * factors["query_specificity"]
    )
    
    return {
        "score": confidence,
        "label": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low",
        "factors": factors
    }
```

Display to user: "Confidence: High (0.87) — Based on 3 Supreme Court judgments and the current statutory text."

#### Layer 7d: Grounded Refinement (Post-Generation)

After the LLM generates a response, run a separate verification pass:

```python
def grounded_refinement(response, retrieved_chunks):
    prompt = f"""You are a legal accuracy auditor. Review this AI-generated legal 
    response and the source documents it was based on.
    
    Response: {response}
    Source documents: {retrieved_chunks}
    
    For each claim in the response:
    1. Is it directly supported by the source documents? (Grounded/Ungrounded)
    2. Is any source mischaracterized? (Accurate/Mischaracterized)
    3. Are there important caveats from the sources that were omitted? (Complete/Incomplete)
    
    Return a revised response that:
    - Removes or flags ungrounded claims
    - Corrects any mischaracterizations  
    - Adds important omitted caveats
    """
    
    refined_response = llm.generate(prompt)
    return refined_response
```

This is based on DO-RAG (May 2025) which showed significant improvement in answer fidelity through post-generation cross-referencing against knowledge graph evidence.

#### Layer 7e: Finetune-RAG — Training the Model to Resist Hallucination

The Finetune-RAG paper (arXiv 2505.10792, Dec 2025) demonstrates a 21.2% improvement in factual accuracy by fine-tuning the generation model to resist misleading retrieved context:

**The insight:** Even with perfect retrieval, the generation model sometimes weaves in incorrect information from tangentially relevant but factually wrong retrieved chunks. Finetune-RAG trains the model to:
- Distinguish between factual and misleading context
- Refuse to generate claims not supported by retrieved documents
- Say "I don't have enough information" instead of hallucinating

**Implementation for your system:**
1. Create a training dataset with real Indian legal queries + correct retrieval + some deliberately wrong retrieval mixed in
2. Fine-tune the generation model to always ground in correct sources and flag/ignore incorrect ones
3. This is a Phase 2 optimization — requires significant labeled data

---

### Phase 8: Quality Assurance & Continuous Improvement

#### Automated Evaluation Pipeline (RAGAS Framework)

Run after every system update and on a weekly sample of real queries:

| Metric | Target | What It Measures |
|--------|--------|-----------------|
| Context Recall | >0.90 | Are relevant documents being retrieved? |
| Context Precision | >0.85 | Are irrelevant documents being filtered out? |
| Faithfulness | >0.95 | Are answers grounded in retrieved context? |
| Answer Relevancy | >0.85 | Do answers address the query? |
| Citation Accuracy | >0.98 | Are cited sources real and correctly attributed? |
| Temporal Accuracy | >0.99 | Is the law current (or correctly dated if historical)? |

#### Human Evaluation Protocol

Before launch and quarterly thereafter:

1. **Recruit 5-10 practicing Indian lawyers** across practice areas (criminal, civil, corporate, tax, constitutional)
2. **Generate 200 test queries** — 40 per practice area, mix of factual/analytical/comparative
3. **Evaluate on four dimensions:**
   - Accuracy (Is the legal information correct?)
   - Completeness (Are all relevant provisions cited?)
   - Recency (Is the law current?)
   - Usefulness (Would this actually help build a case?)
4. **Target: 85%+ accuracy on lawyer evaluation** before public launch

#### Continuous Improvement Loop

```
Weekly:
├── Run RAGAS on sample of real user queries
├── Review flagged hallucinations from Layer 7
├── Monitor retrieval latency and error rates
├── Check for new amendments/judgments not yet ingested

Monthly:
├── Analyze query patterns → identify gaps in knowledge base
├── Re-evaluate chunking boundaries based on retrieval metrics
├── Update fine-tuned embedding model with new data
├── Human evaluation on 50 random queries

Quarterly:
├── Full lawyer evaluation (200 queries)
├── Re-benchmark against commercial tools
├── Review and update knowledge graph schema if needed
└── Publish accuracy report (builds trust with law firm buyers)
```

---

## PART III: IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Weeks 1-6) — Get to a Working Demo

**Goal:** Parse 50 major Central Acts + last 2 years of Supreme Court judgments. Basic hybrid search working. No knowledge graph yet.

| Week | Deliverable |
|------|------------|
| 1-2 | Set up Docling parsing pipeline. Ingest top 50 Central Acts from India Code. Structure-aware chunking for statutes. |
| 3-4 | Ingest SC judgments (2023-2025). Structural chunking for judgments. Set up Qdrant with hybrid search (BM25 + vector). |
| 5-6 | Implement Contextual Retrieval for all chunks. Basic reranking. Build minimal chat interface. First demo to 2-3 lawyers. |

**Cost:** ~$100 (Contextual Retrieval LLM calls) + compute costs
**Risk:** Parser may struggle with some Indian PDF formats. Budget time for debugging.

### Phase 2: Intelligence Layer (Weeks 7-14) — Make It Actually Good

**Goal:** Knowledge graph, Late Chunking, fine-tuned embeddings, QuIM-RAG. This is where it goes from "interesting demo" to "actually useful."

| Week | Deliverable |
|------|------------|
| 7-8 | Build Neo4j knowledge graph. Ingest citation relationships between Acts and judgments. Implement temporal status tracking. |
| 9-10 | Implement Late Chunking. Re-embed all chunks with Late Chunking approach. Compare retrieval metrics before/after. |
| 11-12 | Fine-tune embedding model on Indian legal text. Create training dataset from Indian Kanoon + statute Q&A pairs. |
| 13-14 | Implement QuIM-RAG question generation. Add RAPTOR summary trees for all 50 Acts. |

**Cost:** ~$200 (LLM calls for QuIM + RAPTOR) + GPU cost for fine-tuning (~$50 for 8 hours on A100)
**Risk:** Fine-tuning may require iteration. Budget 2 additional weeks if first results are poor.

### Phase 3: Trust Layer (Weeks 15-20) — Make It Trustworthy

**Goal:** Full hallucination mitigation pipeline. Lawyer evaluation. Production hardening.

| Week | Deliverable |
|------|------------|
| 15-16 | Implement citation verification (Layer 7a) and temporal consistency checking (Layer 7b). |
| 17-18 | Implement confidence scoring (Layer 7c) and grounded refinement (Layer 7d). RAGAS evaluation pipeline. |
| 19-20 | Recruit 5 lawyers for evaluation. Run 200-query test. Fix issues found. Achieve 85%+ accuracy target. |

**Cost:** ~$500-1000 (lawyer compensation for evaluation)
**Risk:** Lawyers may surface issues you didn't anticipate. This is good — better to find them now than after launch.

### Phase 4: Scale (Weeks 21-30) — Production Launch

**Goal:** Scale to all Central Acts (800+), all High Courts, agentic acquisition, continuous sync. Launch to paying customers.

| Week | Deliverable |
|------|------------|
| 21-24 | Scale to all Central Acts + all High Court judgments (last 5 years). Build agentic source discovery and change detection. |
| 25-28 | Multi-tenancy for firms (each firm's uploaded docs isolated). Continuous sync for new judgments/amendments. |
| 29-30 | Production deployment. Monitoring dashboards. Public launch. |

---

## PART IV: COST PROJECTIONS

### One-Time Ingestion Costs (For Full Indian Legal Corpus)

| Component | Estimated Volume | Cost |
|-----------|-----------------|------|
| Docling parsing (self-hosted) | ~1M pages | $0 (compute: ~$50 for GPU time) |
| Contextual Retrieval (Claude Haiku) | ~500K chunks | ~$500 |
| QuIM-RAG question generation | ~500K chunks × 5 questions | ~$200 |
| RAPTOR summaries | ~800 Acts + major judgment sets | ~$300 |
| Fine-tuned embedding inference | ~500K chunks + 2.5M questions | ~$100 (GPU time) |
| Neo4j (self-hosted Community Edition) | ~5M nodes, ~20M relationships | $0 |
| Qdrant (self-hosted) | ~3M vectors | $0 |
| **Total one-time ingestion** | | **~$1,150** |

### Monthly Operating Costs

| Component | Cost |
|-----------|------|
| GPU server (for embedding inference + parsing new docs) | $100-200 |
| LLM API (for generation, hallucination mitigation) | $200-500 (scales with usage) |
| Cloud server (Qdrant + Neo4j + application) | $100-300 |
| Incremental ingestion (new judgments, amendments) | $50-100 |
| **Total monthly** | **$450-1,100** |

At ₹10,000/month per individual lawyer, you need **5-12 paying customers to break even on infrastructure.**

---

## PART V: KEY RESEARCH PAPERS — ANNOTATED READING LIST

### Must-Read (Directly Shaped This Architecture)

1. **"Contextual Retrieval"** (Anthropic, Sep 2024) — anthropic.com/news/contextual-retrieval
   *Why:* The 67% retrieval failure reduction technique. Core of your BM25 enrichment strategy.

2. **"Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"** (Jina AI, arXiv 2409.04701, Jul 2025 revision)
   *Why:* Complements Contextual Retrieval for embeddings. Zero additional training required.

3. **"An Ontology-Driven Graph RAG for Legal Norms: A Structural, Temporal, and Deterministic Approach"** (arXiv 2505.00039, Sep 2025)
   *Why:* The SAT-Graph RAG paper. Directly informs your knowledge graph schema and temporal query capability.

4. **"Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models"** (Stanford, Journal of Empirical Legal Studies, 2025)
   *Why:* The sobering reality check. Legal RAG tools hallucinate 17-33%. Your hallucination mitigation layer exists because of this paper.

5. **"Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination"** (arXiv 2505.10792, Dec 2025)
   *Why:* 21.2% factual accuracy improvement by fine-tuning the generation model. Phase 2 optimization for your system.

6. **"Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion"** (IBM Research, AAAI 2025)
   *Why:* Your primary parser. Understanding its architecture helps you optimize configuration for Indian legal documents.

### Important (Informs Design Decisions)

7. **"RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"** (ICLR 2024, arXiv 2401.18059)
   *Why:* Hierarchical summary trees for multi-hop reasoning. 20% improvement on complex Q&A benchmarks.

8. **"Benchmarking KG-based RAG Systems: A Case Study of Legal Documents"** (CEUR Workshop, 2025)
   *Why:* Proves hybrid (graph + vector) outperforms either alone. Validates your architecture choice.

9. **"Max-Min Semantic Chunking for RAG"** (Springer, Discover Computing, Jun 2025)
   *Why:* Best semantic chunking algorithm for documents without clean structural markers.

10. **"Enhancing the Precision and Interpretability of RAG in Legal Technology: A Survey"** (IEEE Access, 2025, DOI: 10.1109/ACCESS.2025.3550145)
    *Why:* Comprehensive survey of legal RAG failure points and mitigation techniques. Reference for your QA pipeline.

11. **"Fine-tuning Open Source Embedding Models for Improving Retrieval in Legal-RAG"** (Medium, Jun 2025, by axon_dendrite)
    *Why:* Practical guide to fine-tuning BGE on Indian SEBI regulatory text. 16% improvement. Directly applicable to your embedding fine-tuning plan.

12. **"A Comparative Study of PDF Parsing Tools Across Diverse Document Categories"** (arXiv 2410.09871, Oct 2024)
    *Why:* Benchmark of 10 parsers across legal documents. Informs your parser selection.

13. **"Evaluating Chunking Strategies for Retrieval"** (Chroma Research, 2025)
    *Why:* Introduces token-wise IoU metric. Helps you evaluate your chunking strategy rigorously.

14. **"Reconstructing Context: Late Chunking vs. Contextual Retrieval"** (ECIR 2025 Workshop)
    *Why:* Direct comparison of the two techniques. Validates your decision to use both.

---

## PART VI: COMPETITIVE POSITIONING STATEMENT

When you pitch this to investors or law firms, here's what makes this pipeline distinctive:

**"We combine six techniques that no other Indian legal AI system currently uses together:**

1. **Structure-aware chunking** that respects how Indian statutes and judgments are actually organized — not naive 500-token splitting
2. **Dual context enrichment** (Late Chunking + Contextual Retrieval) for the industry's best retrieval accuracy — a 67%+ reduction in retrieval failures
3. **A legal knowledge graph** that tracks amendments, citations, and temporal validity — so we never cite repealed law
4. **Fine-tuned embeddings** specifically trained on Indian legal terminology and cross-references
5. **Pre-generated question matching** (QuIM-RAG) that bridges the semantic gap between how lawyers ask questions and how law is written
6. **A four-layer hallucination mitigation system** including citation verification, temporal checking, confidence scoring, and post-generation grounded refinement

**The result: a system that a lawyer can cite in court with confidence."**
