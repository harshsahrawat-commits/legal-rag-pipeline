from __future__ import annotations

from src.chunking._metadata_builder import MetadataBuilder, _parse_date
from src.chunking._models import ChunkStrategy, ChunkType, CourtHierarchy
from src.parsing._models import ParsedDocument, SectionLevel


class TestBuildSourceInfo:
    def test_builds_from_statute(self, sample_statute_doc: ParsedDocument):
        mb = MetadataBuilder()
        si = mb.build_source_info(sample_statute_doc)
        assert si.source_name == "indian_kanoon"
        assert "raw" in si.url

    def test_builds_from_judgment(self, sample_judgment_doc: ParsedDocument):
        mb = MetadataBuilder()
        si = mb.build_source_info(sample_judgment_doc)
        assert si.scraped_at == sample_judgment_doc.parsed_at


class TestBuildStatuteMetadata:
    def test_basic(self, sample_statute_doc: ParsedDocument):
        mb = MetadataBuilder()
        sm = mb.build_statute_metadata(sample_statute_doc, section_number="10")
        assert sm.act_name == "Indian Contract Act"
        assert sm.act_number == "9 of 1872"
        assert sm.section_number == "10"

    def test_missing_act_name_falls_back_to_title(self, sample_statute_doc: ParsedDocument):
        doc = sample_statute_doc.model_copy(update={"act_name": None})
        mb = MetadataBuilder()
        sm = mb.build_statute_metadata(doc)
        assert sm.act_name == "The Indian Contract Act, 1872"

    def test_chapter_and_part(self, sample_statute_doc: ParsedDocument):
        mb = MetadataBuilder()
        sm = mb.build_statute_metadata(
            sample_statute_doc,
            chapter="I",
            part="A",
            schedule="First Schedule",
        )
        assert sm.chapter == "I"
        assert sm.part == "A"
        assert sm.schedule == "First Schedule"


class TestBuildJudgmentMetadata:
    def test_basic(self, sample_judgment_doc: ParsedDocument):
        mb = MetadataBuilder()
        jm = mb.build_judgment_metadata(sample_judgment_doc)
        assert jm.case_citation == "AIR 2024 SC 1500"
        assert jm.court == "Supreme Court of India"
        assert jm.court_level == CourtHierarchy.SUPREME_COURT

    def test_parties_split(self, sample_judgment_doc: ParsedDocument):
        mb = MetadataBuilder()
        jm = mb.build_judgment_metadata(sample_judgment_doc)
        assert jm.parties_petitioner == "State of Maharashtra"
        assert jm.parties_respondent == "Rajesh Kumar"

    def test_missing_court(self, sample_judgment_doc: ParsedDocument):
        doc = sample_judgment_doc.model_copy(update={"court": None})
        mb = MetadataBuilder()
        jm = mb.build_judgment_metadata(doc)
        assert jm.court == "Unknown Court"

    def test_no_parties(self, sample_judgment_doc: ParsedDocument):
        doc = sample_judgment_doc.model_copy(update={"parties": None})
        mb = MetadataBuilder()
        jm = mb.build_judgment_metadata(doc)
        assert jm.parties_petitioner is None
        assert jm.parties_respondent is None


class TestBuildContentMetadata:
    def test_extracts_section_citations(self):
        mb = MetadataBuilder()
        text = "Section 302 of the Indian Penal Code, 1860 provides punishment."
        cm = mb.build_content_metadata(text)
        assert any("302" in s for s in cm.sections_cited)

    def test_extracts_act_citations(self):
        mb = MetadataBuilder()
        text = "The Indian Penal Code, 1860 was enacted by the British."
        cm = mb.build_content_metadata(text)
        assert any("Indian Penal Code" in a for a in cm.acts_cited)

    def test_extracts_case_citations(self):
        mb = MetadataBuilder()
        text = "In K.M. Nanavati v. State of Maharashtra, AIR 1962 SC 605, this Court held."
        cm = mb.build_content_metadata(text)
        assert any("AIR 1962 SC 605" in c for c in cm.cases_cited)

    def test_detects_hindi(self):
        mb = MetadataBuilder()
        text = "भारतीय दंड संहिता की धारा ३०२ के अंतर्गत दंडनीय अपराध है जो कि हत्या का प्रावधान करती है"
        cm = mb.build_content_metadata(text)
        assert cm.has_hindi is True

    def test_english_only(self):
        mb = MetadataBuilder()
        text = "All agreements are contracts if made by free consent."
        cm = mb.build_content_metadata(text)
        assert cm.has_hindi is False
        assert cm.language == "en"

    def test_no_duplicates(self):
        mb = MetadataBuilder()
        text = "Section 302 IPC and again Section 302 IPC are referenced."
        cm = mb.build_content_metadata(text)
        sec_302_refs = [s for s in cm.sections_cited if "302" in s]
        assert len(sec_302_refs) == 1

    def test_scc_case_citation(self):
        mb = MetadataBuilder()
        text = "as held in (2024) 3 SCC 456."
        cm = mb.build_content_metadata(text)
        assert any("SCC" in c for c in cm.cases_cited)


class TestBuildIngestionMetadata:
    def test_basic(self, sample_statute_doc: ParsedDocument):
        mb = MetadataBuilder()
        im = mb.build_ingestion_metadata(sample_statute_doc, ChunkStrategy.STRUCTURE_BOUNDARY)
        assert im.chunk_strategy == ChunkStrategy.STRUCTURE_BOUNDARY
        assert im.parser == "html_indian_kanoon"
        assert im.requires_manual_review is False

    def test_degraded_scan(self, sample_degraded_scan_doc: ParsedDocument):
        mb = MetadataBuilder()
        im = mb.build_ingestion_metadata(
            sample_degraded_scan_doc,
            ChunkStrategy.PAGE_LEVEL,
            requires_manual_review=True,
        )
        assert im.requires_manual_review is True
        assert im.ocr_confidence == 0.65


class TestClassifyChunkType:
    def test_section_to_statutory_text(self):
        assert MetadataBuilder.classify_chunk_type(SectionLevel.SECTION) == ChunkType.STATUTORY_TEXT

    def test_definition(self):
        assert MetadataBuilder.classify_chunk_type(SectionLevel.DEFINITION) == ChunkType.DEFINITION

    def test_facts(self):
        assert MetadataBuilder.classify_chunk_type(SectionLevel.FACTS) == ChunkType.FACTS

    def test_holding(self):
        assert MetadataBuilder.classify_chunk_type(SectionLevel.HOLDING) == ChunkType.HOLDING

    def test_schedule(self):
        assert (
            MetadataBuilder.classify_chunk_type(SectionLevel.SCHEDULE) == ChunkType.SCHEDULE_ENTRY
        )

    def test_proviso(self):
        assert MetadataBuilder.classify_chunk_type(SectionLevel.PROVISO) == ChunkType.PROVISO


class TestClassifyCourtHierarchy:
    def test_supreme_court(self):
        assert (
            MetadataBuilder.classify_court_hierarchy("Supreme Court of India")
            == CourtHierarchy.SUPREME_COURT
        )

    def test_high_court(self):
        assert (
            MetadataBuilder.classify_court_hierarchy("High Court of Bombay")
            == CourtHierarchy.HIGH_COURT
        )

    def test_district_court(self):
        assert (
            MetadataBuilder.classify_court_hierarchy("District Court of Delhi")
            == CourtHierarchy.DISTRICT_COURT
        )

    def test_tribunal(self):
        assert (
            MetadataBuilder.classify_court_hierarchy("National Green Tribunal")
            == CourtHierarchy.TRIBUNAL
        )

    def test_unknown_defaults_to_district(self):
        assert (
            MetadataBuilder.classify_court_hierarchy("Some Unknown Body")
            == CourtHierarchy.DISTRICT_COURT
        )


class TestParseDate:
    def test_indian_format(self):
        d = _parse_date("15 March, 2024")
        assert d is not None
        assert d.year == 2024 and d.month == 3 and d.day == 15

    def test_india_code_format(self):
        d = _parse_date("6-Feb-2012")
        assert d is not None
        assert d.year == 2012

    def test_iso_format(self):
        d = _parse_date("2024-03-15")
        assert d is not None

    def test_invalid_returns_none(self):
        assert _parse_date("not a date") is None
