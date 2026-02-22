from __future__ import annotations

from datetime import datetime

from src.acquisition._models import (
    AcquisitionResult,
    ContentFormat,
    CrawlRecord,
    CrawlState,
    DiscoveredDocument,
    DocumentFlag,
    DocumentType,
    FlagSeverity,
    FlagType,
    PreliminaryMetadata,
    RawDocument,
    ScrapedContent,
    SourceType,
)


class TestEnums:
    def test_document_type_values(self):
        assert DocumentType.STATUTE.value == "statute"
        assert DocumentType.JUDGMENT.value == "judgment"

    def test_source_type_values(self):
        assert SourceType.INDIAN_KANOON.value == "indian_kanoon"
        assert SourceType.INDIA_CODE.value == "india_code"

    def test_content_format_values(self):
        assert ContentFormat.HTML.value == "html"
        assert ContentFormat.PDF.value == "pdf"

    def test_flag_types(self):
        assert FlagType.SCANNED_PDF.value == "scanned_pdf"
        assert FlagType.REGIONAL_LANGUAGE.value == "regional_language"


class TestCrawlRecord:
    def test_create_and_serialize(self):
        record = CrawlRecord(
            url="https://indiankanoon.org/doc/123/",
            content_hash="abc123",
            file_path="data/raw/indian_kanoon/123.html",
            scraped_at=datetime(2024, 1, 1),
            document_type=DocumentType.JUDGMENT,
        )
        data = record.model_dump()
        assert data["url"] == "https://indiankanoon.org/doc/123/"
        assert data["document_type"] == "judgment"

    def test_roundtrip_json(self):
        record = CrawlRecord(
            url="https://example.com/doc/1",
            content_hash="hash1",
            file_path="data/raw/test/1.html",
            scraped_at=datetime(2024, 6, 15),
        )
        json_str = record.model_dump_json()
        restored = CrawlRecord.model_validate_json(json_str)
        assert restored.url == record.url
        assert restored.content_hash == record.content_hash


class TestCrawlState:
    def test_empty_state(self):
        state = CrawlState(source_type=SourceType.INDIAN_KANOON)
        assert state.records == {}
        assert state.last_run is None

    def test_state_with_records(self):
        state = CrawlState(
            source_type=SourceType.INDIA_CODE,
            last_run=datetime(2024, 1, 1),
            records={
                "url1": CrawlRecord(
                    url="url1",
                    content_hash="h1",
                    file_path="f1",
                    scraped_at=datetime(2024, 1, 1),
                ),
            },
        )
        assert len(state.records) == 1
        assert "url1" in state.records


class TestDiscoveredDocument:
    def test_defaults(self):
        doc = DiscoveredDocument(
            url="https://example.com/doc/1", source_type=SourceType.INDIAN_KANOON
        )
        assert doc.is_new is True
        assert doc.content_hash_changed is False


class TestScrapedContent:
    def test_create(self):
        sc = ScrapedContent(
            url="https://example.com",
            content="<html>test</html>",
            content_format=ContentFormat.HTML,
            content_hash="abc",
            status_code=200,
        )
        assert sc.content_format == ContentFormat.HTML


class TestPreliminaryMetadata:
    def test_all_optional(self):
        meta = PreliminaryMetadata()
        assert meta.title is None
        assert meta.act_name is None

    def test_partial(self):
        meta = PreliminaryMetadata(title="IPC Section 302", act_name="Indian Penal Code")
        assert meta.title == "IPC Section 302"


class TestDocumentFlag:
    def test_create(self):
        flag = DocumentFlag(
            flag_type=FlagType.SCANNED_PDF,
            message="Document appears to be a scanned image",
            severity=FlagSeverity.WARNING,
        )
        assert flag.severity == FlagSeverity.WARNING


class TestRawDocument:
    def test_create_with_defaults(self):
        doc = RawDocument(
            url="https://indiankanoon.org/doc/123/",
            source_type=SourceType.INDIAN_KANOON,
            content_format=ContentFormat.HTML,
            raw_content_path="data/raw/indian_kanoon/123.html",
        )
        assert doc.document_id is not None
        assert doc.document_type is None
        assert doc.flags == []

    def test_full_document(self):
        doc = RawDocument(
            url="https://indiankanoon.org/doc/456/",
            source_type=SourceType.INDIAN_KANOON,
            content_format=ContentFormat.HTML,
            raw_content_path="data/raw/indian_kanoon/456.html",
            document_type=DocumentType.JUDGMENT,
            preliminary_metadata=PreliminaryMetadata(
                title="State vs Accused",
                court="Supreme Court of India",
                case_citation="AIR 2024 SC 100",
            ),
            flags=[
                DocumentFlag(
                    flag_type=FlagType.SMALL_CONTENT,
                    message="Content under 500 chars",
                    severity=FlagSeverity.INFO,
                ),
            ],
            content_hash="deadbeef",
        )
        json_str = doc.model_dump_json()
        restored = RawDocument.model_validate_json(json_str)
        assert restored.document_type == DocumentType.JUDGMENT
        assert restored.preliminary_metadata.court == "Supreme Court of India"
        assert len(restored.flags) == 1


class TestAcquisitionResult:
    def test_defaults(self):
        result = AcquisitionResult(source_type=SourceType.INDIAN_KANOON)
        assert result.documents_discovered == 0
        assert result.documents_downloaded == 0
        assert result.errors == []
