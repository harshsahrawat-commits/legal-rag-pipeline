from __future__ import annotations

from src.acquisition._models import (
    ContentFormat,
    DocumentType,
    FlagType,
    PreliminaryMetadata,
    RawDocument,
    SourceType,
)
from src.acquisition.agents._legal_review import LegalReviewAgent


def _make_raw_doc(**kwargs):
    defaults = {
        "url": "https://indiankanoon.org/doc/123/",
        "source_type": SourceType.INDIAN_KANOON,
        "content_format": ContentFormat.HTML,
        "raw_content_path": "data/raw/indian_kanoon/123.html",
        "document_type": DocumentType.JUDGMENT,
        "preliminary_metadata": PreliminaryMetadata(title="Test Doc"),
    }
    defaults.update(kwargs)
    return RawDocument(**defaults)


class TestLegalReviewAgent:
    def test_no_flags_for_good_document(self):
        agent = LegalReviewAgent()
        doc = _make_raw_doc()
        content = "x" * 1000

        reviewed = agent.review(doc, content)
        assert len(reviewed.flags) == 0

    def test_flags_small_content(self):
        agent = LegalReviewAgent()
        doc = _make_raw_doc()
        content = "tiny"

        reviewed = agent.review(doc, content)
        flag_types = [f.flag_type for f in reviewed.flags]
        assert FlagType.SMALL_CONTENT in flag_types

    def test_flags_regional_language(self):
        agent = LegalReviewAgent()
        doc = _make_raw_doc()
        # Content mostly non-ASCII (Hindi-like)
        content = "\u0915" * 1000  # Devanagari characters

        reviewed = agent.review(doc, content)
        flag_types = [f.flag_type for f in reviewed.flags]
        assert FlagType.REGIONAL_LANGUAGE in flag_types

    def test_flags_missing_title(self):
        agent = LegalReviewAgent()
        doc = _make_raw_doc(preliminary_metadata=PreliminaryMetadata())
        content = "x" * 1000

        reviewed = agent.review(doc, content)
        flag_types = [f.flag_type for f in reviewed.flags]
        assert FlagType.MISSING_METADATA in flag_types

    def test_flags_scanned_pdf(self):
        agent = LegalReviewAgent()
        doc = _make_raw_doc(content_format=ContentFormat.PDF)
        # Mostly non-alpha content simulating a scanned PDF
        content = "\x00\x01\x02" * 500

        reviewed = agent.review(doc, content)
        flag_types = [f.flag_type for f in reviewed.flags]
        assert FlagType.SCANNED_PDF in flag_types

    def test_deduplicates_flags(self):
        agent = LegalReviewAgent()
        # Start with a flag that will also be detected
        from src.acquisition._models import DocumentFlag, FlagSeverity

        existing_flag = DocumentFlag(
            flag_type=FlagType.SMALL_CONTENT,
            message="Content is only 4 characters",
            severity=FlagSeverity.WARNING,
        )
        doc = _make_raw_doc(flags=[existing_flag])
        content = "tiny"

        reviewed = agent.review(doc, content)
        # Should not have duplicate small_content flags with same message
        small_flags = [f for f in reviewed.flags if f.flag_type == FlagType.SMALL_CONTENT]
        messages = [f.message for f in small_flags]
        assert len(messages) == len(set(messages))
