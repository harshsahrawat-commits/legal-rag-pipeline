from __future__ import annotations

from uuid import uuid4

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._models import ChunkingSettings, ChunkStrategy, ChunkType
from src.chunking._token_counter import TokenCounter
from src.chunking.chunkers._statute_boundary import (
    StatuteBoundaryChunker,
    _count_sections,
    _flatten_section_text,
    _group_children_with_attachments,
)
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    QualityReport,
    SectionLevel,
)


def _make_statute(sections: list[ParsedSection], **overrides) -> ParsedDocument:
    defaults = {
        "document_id": uuid4(),
        "source_type": SourceType.INDIAN_KANOON,
        "document_type": DocumentType.STATUTE,
        "content_format": ContentFormat.HTML,
        "raw_text": "dummy raw text",
        "sections": sections,
        "act_name": "Test Act",
        "act_number": "1 of 2024",
        "parser_used": ParserType.HTML_INDIAN_KANOON,
        "quality": QualityReport(overall_score=0.9, passed=True),
        "raw_content_path": "data/raw/test.html",
    }
    defaults.update(overrides)
    return ParsedDocument(**defaults)


class TestCanChunk:
    def test_statute_with_enough_sections(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_statute_doc) is True

    def test_judgment_returns_false(self, sample_judgment_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        assert chunker.can_chunk(sample_judgment_doc) is False

    def test_statute_too_few_sections(self):
        sections = [
            ParsedSection(id="s1", level=SectionLevel.SECTION, number="1", text="Short."),
        ]
        doc = _make_statute(sections)
        chunker = StatuteBoundaryChunker(
            ChunkingSettings(min_section_count_statute=3), TokenCounter()
        )
        assert chunker.can_chunk(doc) is False


class TestStrategy:
    def test_strategy_value(self):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        assert chunker.strategy == ChunkStrategy.STRUCTURE_BOUNDARY


class TestChunkBasic:
    def test_preamble_gets_own_chunk(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        preamble_chunks = [c for c in chunks if "define and amend" in c.text]
        assert len(preamble_chunks) >= 1

    def test_each_section_becomes_chunk(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        # Should have chunks for: preamble, s1, s2, s10
        assert len(chunks) >= 4

    def test_section_text_includes_children(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        # Section 10 should include explanation and proviso text
        s10_chunks = [c for c in chunks if "Section 10" in c.text]
        assert len(s10_chunks) >= 1
        s10_text = s10_chunks[0].text
        assert "Nothing herein contained" in s10_text  # explanation
        assert "Provided that" in s10_text  # proviso

    def test_statute_metadata_populated(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        for c in chunks:
            assert c.statute is not None
            assert c.statute.act_name == "Indian Contract Act"
            assert c.judgment is None

    def test_section_number_in_statute_meta(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        s1_chunks = [c for c in chunks if c.statute and c.statute.section_number == "1"]
        assert len(s1_chunks) >= 1

    def test_document_id_propagated(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        for c in chunks:
            assert c.document_id == sample_statute_doc.document_id

    def test_token_counts_populated(self, sample_statute_doc: ParsedDocument):
        tc = TokenCounter()
        chunker = StatuteBoundaryChunker(ChunkingSettings(), tc)
        chunks = chunker.chunk(sample_statute_doc)
        for c in chunks:
            assert c.token_count > 0
            assert c.token_count == tc.count(c.text)

    def test_unique_chunk_ids(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestDefinitionSection:
    def test_definition_children_classified(self, sample_statute_doc: ParsedDocument):
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(sample_statute_doc)
        # Section 2 has definition children → chunk_type should be DEFINITION
        s2_chunks = [c for c in chunks if "Section 2" in c.text and "Interpretation" in c.text]
        assert len(s2_chunks) >= 1
        assert s2_chunks[0].chunk_type == ChunkType.DEFINITION


class TestScheduleChunking:
    def test_schedule_gets_own_chunk(self):
        sections = [
            ParsedSection(id="s1", level=SectionLevel.SECTION, number="1", text="Section 1."),
            ParsedSection(id="s2", level=SectionLevel.SECTION, number="2", text="Section 2."),
            ParsedSection(id="s3", level=SectionLevel.SECTION, number="3", text="Section 3."),
            ParsedSection(
                id="sched",
                level=SectionLevel.SCHEDULE,
                title="First Schedule",
                text="Table of rates and duties.",
            ),
        ]
        doc = _make_statute(sections)
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        sched_chunks = [c for c in chunks if c.chunk_type == ChunkType.SCHEDULE_ENTRY]
        assert len(sched_chunks) == 1
        assert sched_chunks[0].statute is not None
        assert sched_chunks[0].statute.schedule == "First Schedule"


class TestOversizedSectionSplit:
    def test_splits_at_subsection_boundaries(self):
        # Create a section with many sub-sections that exceeds max_tokens
        children = [
            ParsedSection(
                id=f"ss{i}",
                level=SectionLevel.SUB_SECTION,
                number=f"({i})",
                text=" ".join(f"word{j}" for j in range(80)),
                parent_id="s1",
            )
            for i in range(10)
        ]
        section = ParsedSection(
            id="s1",
            level=SectionLevel.SECTION,
            number="1",
            title="Long Section",
            text="Introduction text.",
            children=children,
        )
        doc = _make_statute(
            [
                section,
                ParsedSection(id="s2", level=SectionLevel.SECTION, number="2", text="X."),
                ParsedSection(id="s3", level=SectionLevel.SECTION, number="3", text="Y."),
            ],
        )
        settings = ChunkingSettings(max_tokens=200)
        chunker = StatuteBoundaryChunker(settings, TokenCounter())
        chunks = chunker.chunk(doc)
        # Should have more than 1 chunk for section 1
        s1_chunks = [c for c in chunks if "Section 1" in c.text]
        assert len(s1_chunks) > 1

    def test_parent_chunk_id_set_on_splits(self):
        children = [
            ParsedSection(
                id=f"ss{i}",
                level=SectionLevel.SUB_SECTION,
                number=f"({i})",
                text=" ".join(f"word{j}" for j in range(100)),
                parent_id="s1",
            )
            for i in range(10)
        ]
        section = ParsedSection(
            id="s1",
            level=SectionLevel.SECTION,
            number="1",
            title="Long Section",
            text="Base text.",
            children=children,
        )
        doc = _make_statute(
            [
                section,
                ParsedSection(id="s2", level=SectionLevel.SECTION, number="2", text="X."),
                ParsedSection(id="s3", level=SectionLevel.SECTION, number="3", text="Y."),
            ],
        )
        settings = ChunkingSettings(max_tokens=200)
        chunker = StatuteBoundaryChunker(settings, TokenCounter())
        chunks = chunker.chunk(doc)
        s1_chunks = [c for c in chunks if "Section 1" in c.text]
        # At least one sub-chunk should have parent_chunk_id
        has_parent = any(c.parent_info.parent_chunk_id is not None for c in s1_chunks)
        assert has_parent

    def test_proviso_stays_with_parent(self):
        children = [
            ParsedSection(
                id="ss1",
                level=SectionLevel.SUB_SECTION,
                number="(1)",
                text=" ".join(f"word{j}" for j in range(80)),
                parent_id="s1",
            ),
            ParsedSection(
                id="prov1",
                level=SectionLevel.PROVISO,
                text="Provided that this proviso attaches to subsection 1.",
                parent_id="s1",
            ),
            ParsedSection(
                id="ss2",
                level=SectionLevel.SUB_SECTION,
                number="(2)",
                text=" ".join(f"word{j}" for j in range(80)),
                parent_id="s1",
            ),
        ]
        section = ParsedSection(
            id="s1",
            level=SectionLevel.SECTION,
            number="1",
            title="Section with Proviso",
            text="Main section text.",
            children=children,
        )
        doc = _make_statute(
            [
                section,
                ParsedSection(id="s2", level=SectionLevel.SECTION, number="2", text="X."),
                ParsedSection(id="s3", level=SectionLevel.SECTION, number="3", text="Y."),
            ],
        )
        settings = ChunkingSettings(max_tokens=200)
        chunker = StatuteBoundaryChunker(settings, TokenCounter())
        chunks = chunker.chunk(doc)
        # Find chunk containing subsection (1)
        ss1_chunk = None
        for c in chunks:
            if "(1)" in c.text and "Section 1" in c.text:
                ss1_chunk = c
                break
        assert ss1_chunk is not None
        # Proviso text should be in the same chunk
        assert "Provided that" in ss1_chunk.text


class TestHelpers:
    def test_flatten_section_text(self):
        section = ParsedSection(
            id="s1",
            level=SectionLevel.SECTION,
            number="1",
            title="Title",
            text="Body text.",
            children=[
                ParsedSection(
                    id="exp",
                    level=SectionLevel.EXPLANATION,
                    text="Explanation text.",
                    parent_id="s1",
                ),
            ],
        )
        result = _flatten_section_text(section, prefix="Act, Section 1")
        assert "Act, Section 1" in result
        assert "1 Title: Body text." in result
        assert "Explanation text." in result

    def test_count_sections(self):
        sections = [
            ParsedSection(
                id="ch1",
                level=SectionLevel.CHAPTER,
                text="",
                children=[
                    ParsedSection(id="s1", level=SectionLevel.SECTION, number="1", text="A"),
                    ParsedSection(id="s2", level=SectionLevel.SECTION, number="2", text="B"),
                ],
            ),
            ParsedSection(id="s3", level=SectionLevel.SECTION, number="3", text="C"),
        ]
        assert _count_sections(sections) == 3

    def test_group_children_with_attachments(self):
        children = [
            ParsedSection(id="ss1", level=SectionLevel.SUB_SECTION, text="Sub 1"),
            ParsedSection(id="prov1", level=SectionLevel.PROVISO, text="Proviso"),
            ParsedSection(id="ss2", level=SectionLevel.SUB_SECTION, text="Sub 2"),
        ]
        groups = _group_children_with_attachments(children)
        assert len(groups) == 2
        assert len(groups[0]) == 2  # ss1 + proviso
        assert len(groups[1]) == 1  # ss2

    def test_group_children_empty(self):
        assert _group_children_with_attachments([]) == []

    def test_group_children_all_attached(self):
        children = [
            ParsedSection(id="prov1", level=SectionLevel.PROVISO, text="Prov 1"),
            ParsedSection(id="exp1", level=SectionLevel.EXPLANATION, text="Exp 1"),
        ]
        groups = _group_children_with_attachments(children)
        # Both attach to a single group (orphan proviso starts group)
        assert len(groups) == 1
        assert len(groups[0]) == 2


class TestChapterMetadata:
    def test_chapter_tracked_in_metadata(self):
        sections = [
            ParsedSection(
                id="ch1",
                level=SectionLevel.CHAPTER,
                number="I",
                title="Preliminary",
                text="",
                children=[
                    ParsedSection(
                        id="s1", level=SectionLevel.SECTION, number="1", text="Section 1."
                    ),
                    ParsedSection(
                        id="s2", level=SectionLevel.SECTION, number="2", text="Section 2."
                    ),
                    ParsedSection(
                        id="s3", level=SectionLevel.SECTION, number="3", text="Section 3."
                    ),
                ],
            ),
        ]
        doc = _make_statute(sections)
        chunker = StatuteBoundaryChunker(ChunkingSettings(), TokenCounter())
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.statute is not None
            assert c.statute.chapter == "I"
