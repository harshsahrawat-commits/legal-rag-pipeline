from __future__ import annotations

from uuid import uuid4

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._models import ChunkingSettings, ChunkStrategy, ChunkType
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._judgment_structural import JudgmentStructuralChunker
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    QualityReport,
    SectionLevel,
)


def _make_judgment(sections: list[ParsedSection], **overrides) -> ParsedDocument:
    defaults = {
        "document_id": uuid4(),
        "source_type": SourceType.INDIAN_KANOON,
        "document_type": DocumentType.JUDGMENT,
        "content_format": ContentFormat.HTML,
        "raw_text": "dummy",
        "sections": sections,
        "title": "State vs Accused",
        "court": "Supreme Court of India",
        "case_citation": "AIR 2024 SC 100",
        "parties": "State vs Accused",
        "parser_used": ParserType.HTML_INDIAN_KANOON,
        "quality": QualityReport(overall_score=0.9, passed=True),
        "raw_content_path": "data/raw/test.html",
    }
    defaults.update(overrides)
    return ParsedDocument(**defaults)


class TestCanChunk:
    def test_judgment_with_enough_sections(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_judgment_doc) is True

    def test_statute_returns_false(self, sample_statute_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_statute_doc) is False

    def test_judgment_too_few_sections(self):
        sections = [
            ParsedSection(id="h", level=SectionLevel.HEADER, text="Header."),
            ParsedSection(id="p", level=SectionLevel.PARAGRAPH, text="Para."),
        ]
        doc = _make_judgment(sections)
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(doc) is False


class TestStrategy:
    def test_strategy_value(self):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        assert chunker.strategy == ChunkStrategy.JUDGMENT_STRUCTURAL


class TestChunkBasic:
    def test_each_section_becomes_chunk(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        # Header + Facts + Issues + Reasoning + Holding + Order = 6
        assert len(chunks) == 6

    def test_header_chunk_exists(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        header_chunks = [c for c in chunks if "SUPREME COURT" in c.text]
        assert len(header_chunks) >= 1

    def test_judgment_header_chunk_id_set(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        # Find header chunk
        header_chunk = next(c for c in chunks if "SUPREME COURT" in c.text)
        # All other chunks should have judgment_header_chunk_id
        for c in chunks:
            if c.id != header_chunk.id:
                assert c.parent_info.judgment_header_chunk_id == header_chunk.id

    def test_holding_gets_own_chunk(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        holding_chunks = [c for c in chunks if c.chunk_type == ChunkType.HOLDING]
        assert len(holding_chunks) == 1

    def test_chunk_types_correct(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        types = {c.chunk_type for c in chunks}
        assert ChunkType.FACTS in types
        assert ChunkType.ISSUES in types
        assert ChunkType.REASONING in types
        assert ChunkType.HOLDING in types
        assert ChunkType.ORDER in types

    def test_judgment_metadata_populated(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        for c in chunks:
            assert c.judgment is not None
            assert c.judgment.case_citation == "AIR 2024 SC 1500"
            assert c.statute is None

    def test_document_id_propagated(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        for c in chunks:
            assert c.document_id == sample_judgment_doc.document_id

    def test_token_counts_populated(self, sample_judgment_doc: ParsedDocument):
        tc = TokenCounter()
        chunker = JudgmentStructuralChunker(ChunkingSettings(), tc)
        chunks = chunker.chunk(sample_judgment_doc)
        for c in chunks:
            assert c.token_count > 0
            assert c.token_count == tc.count(c.text)

    def test_unique_ids(self, sample_judgment_doc: ParsedDocument):
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_judgment_doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestOversizedSection:
    def test_facts_split_at_paragraphs(self):
        long_facts = "\n\n".join(
            f"Paragraph {i}: " + " ".join(f"word{j}" for j in range(60)) for i in range(10)
        )
        sections = [
            ParsedSection(id="h", level=SectionLevel.HEADER, text="Header text."),
            ParsedSection(id="f", level=SectionLevel.FACTS, title="Facts", text=long_facts),
            ParsedSection(
                id="holding", level=SectionLevel.HOLDING, text="The appeal is dismissed."
            ),
        ]
        doc = _make_judgment(sections)
        settings = ChunkingSettings(max_tokens=200)
        chunker = JudgmentStructuralChunker(settings, TokenCounter())
        chunks = chunker.chunk(doc)
        facts_chunks = [c for c in chunks if c.chunk_type == ChunkType.FACTS]
        assert len(facts_chunks) > 1

    def test_holding_never_split(self):
        long_holding = " ".join(f"word{j}" for j in range(2000))
        sections = [
            ParsedSection(id="h", level=SectionLevel.HEADER, text="Header."),
            ParsedSection(id="f", level=SectionLevel.FACTS, text="Short facts."),
            ParsedSection(id="hold", level=SectionLevel.HOLDING, text=long_holding),
        ]
        doc = _make_judgment(sections)
        settings = ChunkingSettings(max_tokens=200)
        chunker = JudgmentStructuralChunker(settings, TokenCounter())
        chunks = chunker.chunk(doc)
        holding_chunks = [c for c in chunks if c.chunk_type == ChunkType.HOLDING]
        # Holding is never split even if oversized
        assert len(holding_chunks) == 1

    def test_parent_chunk_id_on_splits(self):
        long_text = "\n\n".join(
            f"Para {i}: " + " ".join(f"w{j}" for j in range(80)) for i in range(10)
        )
        sections = [
            ParsedSection(id="h", level=SectionLevel.HEADER, text="Header."),
            ParsedSection(id="r", level=SectionLevel.REASONING, text=long_text),
            ParsedSection(id="hold", level=SectionLevel.HOLDING, text="Dismissed."),
        ]
        doc = _make_judgment(sections)
        settings = ChunkingSettings(max_tokens=200)
        chunker = JudgmentStructuralChunker(settings, TokenCounter())
        chunks = chunker.chunk(doc)
        reasoning_chunks = [c for c in chunks if c.chunk_type == ChunkType.REASONING]
        assert len(reasoning_chunks) > 1
        # At least one should have parent_chunk_id
        has_parent = any(c.parent_info.parent_chunk_id is not None for c in reasoning_chunks)
        assert has_parent


class TestDissent:
    def test_dissent_gets_own_chunk(self):
        sections = [
            ParsedSection(id="h", level=SectionLevel.HEADER, text="Header."),
            ParsedSection(id="f", level=SectionLevel.FACTS, text="Facts."),
            ParsedSection(id="hold", level=SectionLevel.HOLDING, text="Allowed."),
            ParsedSection(
                id="d",
                level=SectionLevel.DISSENT,
                title="Dissenting Opinion",
                text="I am unable to agree with the majority.",
            ),
        ]
        doc = _make_judgment(sections)
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        dissent_chunks = [c for c in chunks if c.chunk_type == ChunkType.DISSENT]
        assert len(dissent_chunks) == 1
        assert "unable to agree" in dissent_chunks[0].text


class TestNoHeader:
    def test_works_without_header(self):
        sections = [
            ParsedSection(id="f", level=SectionLevel.FACTS, text="Facts."),
            ParsedSection(id="hold", level=SectionLevel.HOLDING, text="Dismissed."),
        ]
        doc = _make_judgment(sections)
        chunker = JudgmentStructuralChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        # No header → no judgment_header_chunk_id
        for c in chunks:
            assert c.parent_info.judgment_header_chunk_id is None
