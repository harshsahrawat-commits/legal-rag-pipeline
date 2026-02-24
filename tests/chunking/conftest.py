from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003
from uuid import UUID, uuid4

import pytest

from src.acquisition._models import ContentFormat, DocumentType, SourceType
from src.chunking._models import ChunkingConfig, ChunkingSettings
from src.chunking._token_counter import TokenCounter
from src.parsing._models import (
    ParsedDocument,
    ParsedSection,
    ParserType,
    QualityCheckResult,
    QualityReport,
    SectionLevel,
)

# --- Shared fixtures ---

FIXED_DOC_ID = UUID("12345678-1234-1234-1234-123456789abc")
FIXED_TIMESTAMP = datetime(2026, 2, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def token_counter() -> TokenCounter:
    return TokenCounter()


@pytest.fixture()
def chunking_settings(tmp_path: Path) -> ChunkingSettings:
    return ChunkingSettings(
        input_dir=tmp_path / "parsed",
        output_dir=tmp_path / "chunks",
    )


@pytest.fixture()
def chunking_config(chunking_settings: ChunkingSettings) -> ChunkingConfig:
    return ChunkingConfig(settings=chunking_settings)


@pytest.fixture()
def sample_quality_report() -> QualityReport:
    return QualityReport(
        overall_score=0.95,
        passed=True,
        checks=[
            QualityCheckResult(
                check_name="text_completeness",
                passed=True,
                score=0.95,
                details="Adequate text extracted",
            ),
        ],
    )


@pytest.fixture()
def sample_statute_sections() -> list[ParsedSection]:
    """A simple statute tree: Preamble + Chapter with 3 Sections."""
    return [
        ParsedSection(
            id="preamble",
            level=SectionLevel.PREAMBLE,
            text="An Act to define and amend certain parts of the law relating to contracts.",
        ),
        ParsedSection(
            id="ch1",
            level=SectionLevel.CHAPTER,
            number="I",
            title="PRELIMINARY",
            text="",
            children=[
                ParsedSection(
                    id="s1",
                    level=SectionLevel.SECTION,
                    number="1",
                    title="Short title",
                    text="This Act may be called the Indian Contract Act, 1872.",
                    parent_id="ch1",
                ),
                ParsedSection(
                    id="s2",
                    level=SectionLevel.SECTION,
                    number="2",
                    title="Interpretation clause",
                    text=(
                        "In this Act the following words and expressions are used "
                        "in the following senses, unless a contrary intention appears "
                        "from the context:"
                    ),
                    parent_id="ch1",
                    children=[
                        ParsedSection(
                            id="s2_def_a",
                            level=SectionLevel.DEFINITION,
                            number="(a)",
                            text=(
                                "When one person signifies to another his willingness "
                                "to do or to abstain from doing anything, with a view to "
                                "obtaining the assent of that other to such act or abstinence, "
                                "he is said to make a proposal;"
                            ),
                            parent_id="s2",
                        ),
                        ParsedSection(
                            id="s2_def_b",
                            level=SectionLevel.DEFINITION,
                            number="(b)",
                            text=(
                                "When the person to whom the proposal is made signifies "
                                "his assent thereto, the proposal is said to be accepted;"
                            ),
                            parent_id="s2",
                        ),
                    ],
                ),
                ParsedSection(
                    id="s10",
                    level=SectionLevel.SECTION,
                    number="10",
                    title="What agreements are contracts",
                    text=(
                        "All agreements are contracts if they are made by the free "
                        "consent of parties competent to contract, for a lawful "
                        "consideration and with a lawful object, and are not hereby "
                        "expressly declared to be void."
                    ),
                    parent_id="ch1",
                    children=[
                        ParsedSection(
                            id="s10_exp",
                            level=SectionLevel.EXPLANATION,
                            text=(
                                "Nothing herein contained shall be deemed to affect "
                                "any law in force in India, and not hereby expressly "
                                "repealed, by which any contract is required to be made "
                                "in writing or in the presence of witnesses."
                            ),
                            parent_id="s10",
                        ),
                        ParsedSection(
                            id="s10_prov",
                            level=SectionLevel.PROVISO,
                            text=(
                                "Provided that nothing in this section shall apply to "
                                "contracts made before the commencement of this Act."
                            ),
                            parent_id="s10",
                        ),
                    ],
                ),
            ],
        ),
    ]


@pytest.fixture()
def sample_statute_doc(
    sample_quality_report: QualityReport,
    sample_statute_sections: list[ParsedSection],
) -> ParsedDocument:
    """A ParsedDocument representing a statute with structure."""
    return ParsedDocument(
        document_id=FIXED_DOC_ID,
        source_type=SourceType.INDIAN_KANOON,
        document_type=DocumentType.STATUTE,
        content_format=ContentFormat.HTML,
        raw_text=(
            "The Indian Contract Act, 1872\n"
            "An Act to define and amend certain parts of the law.\n"
            "Section 1. Short title. This Act may be called the Indian Contract Act, 1872.\n"
            "Section 2. Interpretation clause. In this Act the following words...\n"
            "Section 10. What agreements are contracts. All agreements are contracts..."
        ),
        sections=sample_statute_sections,
        title="The Indian Contract Act, 1872",
        act_name="Indian Contract Act",
        act_number="9 of 1872",
        year=1872,
        parser_used=ParserType.HTML_INDIAN_KANOON,
        quality=sample_quality_report,
        raw_content_path="data/raw/indian_kanoon/12345678-1234-1234-1234-123456789abc.html",
        parsed_at=FIXED_TIMESTAMP,
    )


@pytest.fixture()
def sample_judgment_sections() -> list[ParsedSection]:
    """A judgment tree: Header, Facts, Issues, Reasoning, Holding, Order."""
    return [
        ParsedSection(
            id="header",
            level=SectionLevel.HEADER,
            title="Case Header",
            text=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Criminal Appeal No. 1234 of 2023\n"
                "State of Maharashtra ... Appellant\n"
                "Versus\n"
                "Rajesh Kumar ... Respondent"
            ),
        ),
        ParsedSection(
            id="facts",
            level=SectionLevel.FACTS,
            title="Facts of the Case",
            text=(
                "The appellant State of Maharashtra filed this appeal challenging "
                "the order of the High Court of Bombay dated 10 January 2023. "
                "The respondent was charged under Section 302 of the Indian Penal "
                "Code, 1860 for the alleged murder of one Suresh Sharma on 5 May 2022."
            ),
        ),
        ParsedSection(
            id="issues",
            level=SectionLevel.ISSUES,
            title="Issues for Consideration",
            text=(
                "1. Whether the High Court was correct in acquitting the respondent?\n"
                "2. Whether the circumstantial evidence was sufficient to prove guilt "
                "beyond reasonable doubt?"
            ),
        ),
        ParsedSection(
            id="reasoning",
            level=SectionLevel.REASONING,
            title="Analysis and Reasoning",
            text=(
                "We have carefully considered the submissions made by learned counsel "
                "for both parties. Section 302 IPC provides that whoever commits murder "
                "shall be punished with death or imprisonment for life. The burden of "
                "proof lies on the prosecution to establish guilt beyond reasonable doubt."
            ),
        ),
        ParsedSection(
            id="holding",
            level=SectionLevel.HOLDING,
            title="Holding",
            text=(
                "For the foregoing reasons, we hold that the High Court was correct "
                "in its assessment. The circumstantial evidence does not form a "
                "complete chain. The appeal is dismissed."
            ),
        ),
        ParsedSection(
            id="order",
            level=SectionLevel.ORDER,
            title="Order",
            text="The appeal is dismissed. No order as to costs.",
        ),
    ]


@pytest.fixture()
def sample_judgment_doc(
    sample_quality_report: QualityReport,
    sample_judgment_sections: list[ParsedSection],
) -> ParsedDocument:
    """A ParsedDocument representing a judgment with structure."""
    return ParsedDocument(
        document_id=FIXED_DOC_ID,
        source_type=SourceType.INDIAN_KANOON,
        document_type=DocumentType.JUDGMENT,
        content_format=ContentFormat.HTML,
        raw_text=(
            "IN THE SUPREME COURT OF INDIA\n"
            "Criminal Appeal No. 1234 of 2023\n"
            "State Of Maharashtra vs Rajesh Kumar\n"
            "FACTS OF THE CASE\n"
            "The appellant State of Maharashtra filed this appeal...\n"
            "ISSUES FOR CONSIDERATION\n"
            "1. Whether the High Court was correct...\n"
            "HOLDING\n"
            "For the foregoing reasons, we hold that..."
        ),
        sections=sample_judgment_sections,
        title="State Of Maharashtra vs Rajesh Kumar",
        court="Supreme Court of India",
        case_citation="AIR 2024 SC 1500",
        parties="State of Maharashtra vs Rajesh Kumar",
        date="15 March, 2024",
        year=2024,
        parser_used=ParserType.HTML_INDIAN_KANOON,
        quality=sample_quality_report,
        raw_content_path="data/raw/indian_kanoon/12345678-1234-1234-1234-123456789abc.html",
        parsed_at=FIXED_TIMESTAMP,
    )


@pytest.fixture()
def sample_degraded_scan_doc(sample_quality_report: QualityReport) -> ParsedDocument:
    """A degraded scan with low OCR confidence."""
    return ParsedDocument(
        document_id=uuid4(),
        source_type=SourceType.INDIA_CODE,
        document_type=DocumentType.STATUTE,
        content_format=ContentFormat.PDF,
        raw_text="Page 1 text.\fPage 2 text.\fPage 3 text.",
        sections=[],
        title="Some Degraded Statute",
        parser_used=ParserType.DOCLING_PDF,
        ocr_applied=True,
        ocr_confidence=0.65,
        quality=sample_quality_report,
        raw_content_path="data/raw/india_code/degraded.pdf",
        parsed_at=FIXED_TIMESTAMP,
    )


@pytest.fixture()
def sample_unstructured_doc(sample_quality_report: QualityReport) -> ParsedDocument:
    """An unstructured circular with no sections."""
    return ParsedDocument(
        document_id=uuid4(),
        source_type=SourceType.INDIAN_KANOON,
        document_type=DocumentType.CIRCULAR,
        content_format=ContentFormat.HTML,
        raw_text=(
            "Government of India\n"
            "Ministry of Finance\n\n"
            "Circular No. 15/2024\n\n"
            "Subject: Guidelines for digital payments.\n\n"
            "All banks are directed to comply with the following guidelines "
            "effective from 1 April 2024. The Reserve Bank of India has mandated "
            "that all UPI transactions above Rs. 1,00,000 require additional verification."
        ),
        sections=[],
        title="Circular No. 15/2024 - Digital Payments Guidelines",
        parser_used=ParserType.HTML_INDIAN_KANOON,
        quality=sample_quality_report,
        raw_content_path="data/raw/indian_kanoon/circular.html",
        parsed_at=FIXED_TIMESTAMP,
    )
