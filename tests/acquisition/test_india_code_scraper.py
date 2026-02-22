from __future__ import annotations

import pytest

from src.acquisition._models import (
    ContentFormat,
    DocumentType,
    ScrapeConfig,
    SourceDefinition,
    SourceType,
)
from src.acquisition._state import CrawlStateStore
from src.acquisition.scrapers._india_code import IndiaCodeScraper, _is_non_latin


@pytest.fixture()
def ic_source_def():
    return SourceDefinition(
        name="India Code",
        source_type=SourceType.INDIA_CODE,
        base_url="https://www.indiacode.nic.in",
        rate_limit_requests_per_second=100.0,
        request_timeout_seconds=10,
        max_retries=1,
        scrape_config=ScrapeConfig(
            max_pages_per_query=3,
            max_documents=10,
        ),
    )


@pytest.fixture()
def ic_scraper(ic_source_def, tmp_state_dir, tmp_output_dir):
    store = CrawlStateStore(tmp_state_dir)
    return IndiaCodeScraper(ic_source_def, store, tmp_output_dir)


# ------------------------------------------------------------------
# Browse listing parsing
# ------------------------------------------------------------------


class TestBrowseListingParsing:
    def test_parse_extracts_entries(self, ic_scraper, sample_india_code_browse_html):
        entries = ic_scraper._parse_browse_listing(
            "https://www.indiacode.nic.in", sample_india_code_browse_html
        )
        assert len(entries) == 3

    def test_parse_extracts_handle(self, ic_scraper, sample_india_code_browse_html):
        entries = ic_scraper._parse_browse_listing(
            "https://www.indiacode.nic.in", sample_india_code_browse_html
        )
        assert entries[0].handle == "1505"
        assert entries[1].handle == "1999"
        assert entries[2].handle == "22501"

    def test_parse_extracts_title(self, ic_scraper, sample_india_code_browse_html):
        entries = ic_scraper._parse_browse_listing(
            "https://www.indiacode.nic.in", sample_india_code_browse_html
        )
        assert entries[0].short_title == "The Indian Penal Code, 1860"
        assert entries[1].short_title == "The Information Technology Act, 2000"

    def test_parse_extracts_act_number(self, ic_scraper, sample_india_code_browse_html):
        entries = ic_scraper._parse_browse_listing(
            "https://www.indiacode.nic.in", sample_india_code_browse_html
        )
        assert entries[0].act_number == "45"
        assert entries[1].act_number == "21"

    def test_parse_extracts_date(self, ic_scraper, sample_india_code_browse_html):
        entries = ic_scraper._parse_browse_listing(
            "https://www.indiacode.nic.in", sample_india_code_browse_html
        )
        assert entries[0].enactment_date == "6-Oct-1860"
        assert entries[1].enactment_date == "9-Jun-2000"

    def test_parse_builds_detail_url(self, ic_scraper, sample_india_code_browse_html):
        entries = ic_scraper._parse_browse_listing(
            "https://www.indiacode.nic.in", sample_india_code_browse_html
        )
        assert entries[0].detail_url == (
            "https://www.indiacode.nic.in/handle/123456789/1505?view_type=browse"
        )

    def test_parse_empty_table(self, ic_scraper, sample_india_code_browse_empty_html):
        entries = ic_scraper._parse_browse_listing(
            "https://www.indiacode.nic.in", sample_india_code_browse_empty_html
        )
        assert entries == []

    def test_parse_skips_header_rows(self, ic_scraper, sample_india_code_browse_html):
        """Header row has <th> not <td>, so it should be skipped."""
        entries = ic_scraper._parse_browse_listing(
            "https://www.indiacode.nic.in", sample_india_code_browse_html
        )
        # Should only have data rows, not the header
        for entry in entries:
            assert entry.short_title != "Short Title"


# ------------------------------------------------------------------
# PDF URL extraction
# ------------------------------------------------------------------


class TestPdfUrlExtraction:
    def test_extracts_english_pdf(self, ic_scraper, sample_india_code_detail_html):
        url = ic_scraper._extract_pdf_url(sample_india_code_detail_html)
        assert url == "/bitstream/123456789/1999/1/A2000-21%20%281%29.pdf"

    def test_skips_hindi_pdf(self, ic_scraper, sample_india_code_detail_html):
        url = ic_scraper._extract_pdf_url(sample_india_code_detail_html)
        # Should return English PDF, not Hindi
        assert "H2000-21" not in (url or "")

    def test_returns_none_when_no_pdf(self, ic_scraper, sample_india_code_detail_no_pdf_html):
        url = ic_scraper._extract_pdf_url(sample_india_code_detail_no_pdf_html)
        assert url is None


# ------------------------------------------------------------------
# Discovery (with mocked HTTP)
# ------------------------------------------------------------------


class TestIndiaCodeDiscovery:
    async def test_discover_urls_paginates(
        self,
        ic_scraper,
        sample_india_code_browse_html,
        sample_india_code_browse_empty_html,
    ):
        """discover_urls should return detail page URLs from browse listing."""
        from unittest.mock import AsyncMock

        from src.acquisition._models import ScrapedContent

        responses = [
            ScrapedContent(
                url="https://www.indiacode.nic.in/browse?offset=0",
                content=sample_india_code_browse_html,
                content_format=ContentFormat.HTML,
                content_hash="abc123",
                status_code=200,
            ),
            ScrapedContent(
                url="https://www.indiacode.nic.in/browse?offset=100",
                content=sample_india_code_browse_empty_html,
                content_format=ContentFormat.HTML,
                content_hash="def456",
                status_code=200,
            ),
        ]

        mock_client = AsyncMock()
        mock_client.fetch = AsyncMock(side_effect=responses)

        urls = await ic_scraper.discover_urls(mock_client)

        assert len(urls) == 3
        assert "1505" in urls[0]
        assert "1999" in urls[1]

    async def test_discover_respects_max_documents(
        self, ic_source_def, tmp_state_dir, tmp_output_dir, sample_india_code_browse_html
    ):
        """Should stop after max_documents is reached."""
        from unittest.mock import AsyncMock

        from src.acquisition._models import ScrapedContent

        ic_source_def.scrape_config.max_documents = 2
        store = CrawlStateStore(tmp_state_dir)
        scraper = IndiaCodeScraper(ic_source_def, store, tmp_output_dir)

        # The browse page has 3 entries but max_documents=2, so it should stop at 2
        mock_client = AsyncMock()
        mock_client.fetch = AsyncMock(
            return_value=ScrapedContent(
                url="https://www.indiacode.nic.in/browse",
                content=sample_india_code_browse_html,
                content_format=ContentFormat.HTML,
                content_hash="abc123",
                status_code=200,
            )
        )

        urls = await scraper.discover_urls(mock_client)
        assert len(urls) == 2
        # Should only fetch one page since max_documents was reached within it
        assert mock_client.fetch.call_count == 1

    async def test_discover_populates_browse_metadata(
        self,
        ic_scraper,
        sample_india_code_browse_html,
        sample_india_code_browse_empty_html,
    ):
        """Browse metadata dict should be populated during discovery."""
        from unittest.mock import AsyncMock

        from src.acquisition._models import ScrapedContent

        responses = [
            ScrapedContent(
                url="https://www.indiacode.nic.in/browse?offset=0",
                content=sample_india_code_browse_html,
                content_format=ContentFormat.HTML,
                content_hash="abc",
                status_code=200,
            ),
            ScrapedContent(
                url="https://www.indiacode.nic.in/browse?offset=100",
                content=sample_india_code_browse_empty_html,
                content_format=ContentFormat.HTML,
                content_hash="def",
                status_code=200,
            ),
        ]

        mock_client = AsyncMock()
        mock_client.fetch = AsyncMock(side_effect=responses)

        urls = await ic_scraper.discover_urls(mock_client)

        assert len(ic_scraper._browse_metadata) == len(urls)
        entry = ic_scraper._browse_metadata[urls[0]]
        assert entry.short_title == "The Indian Penal Code, 1860"

    async def test_discover_stops_on_empty_page(
        self, ic_scraper, sample_india_code_browse_html, sample_india_code_browse_empty_html
    ):
        """Should stop pagination when an empty page is encountered."""
        from unittest.mock import AsyncMock

        from src.acquisition._models import ScrapedContent

        responses = [
            ScrapedContent(
                url="https://www.indiacode.nic.in/browse?offset=0",
                content=sample_india_code_browse_html,
                content_format=ContentFormat.HTML,
                content_hash="abc",
                status_code=200,
            ),
            ScrapedContent(
                url="https://www.indiacode.nic.in/browse?offset=100",
                content=sample_india_code_browse_empty_html,
                content_format=ContentFormat.HTML,
                content_hash="def",
                status_code=200,
            ),
        ]

        mock_client = AsyncMock()
        mock_client.fetch = AsyncMock(side_effect=responses)

        urls = await ic_scraper.discover_urls(mock_client)

        assert len(urls) == 3  # Only from the first page
        assert mock_client.fetch.call_count == 2


# ------------------------------------------------------------------
# Metadata extraction
# ------------------------------------------------------------------


class TestIndiaCodeMetadata:
    def test_extract_from_browse_cache(self, ic_scraper, sample_india_code_detail_html):
        """Metadata should come from browse cache + PDF URL from detail page."""
        from src.acquisition.scrapers._india_code import _BrowseEntry

        url = "https://www.indiacode.nic.in/handle/123456789/1999?view_type=browse"
        ic_scraper._browse_metadata[url] = _BrowseEntry(
            handle="1999",
            detail_url=url,
            short_title="The Information Technology Act, 2000",
            act_number="21",
            enactment_date="9-Jun-2000",
        )

        meta = ic_scraper.extract_metadata(url, sample_india_code_detail_html)

        assert meta.title == "The Information Technology Act, 2000"
        assert meta.act_name == "The Information Technology Act, 2000"
        assert meta.act_number == "21"
        assert meta.year == 2000
        assert meta.date == "9-Jun-2000"
        assert meta.download_url is not None
        assert "A2000-21" in meta.download_url

    def test_extract_missing_cache_returns_empty(self, ic_scraper):
        """When URL is not in browse cache, metadata should be mostly empty."""
        meta = ic_scraper.extract_metadata(
            "https://www.indiacode.nic.in/handle/123456789/9999", "<html></html>"
        )
        assert meta.title is None
        assert meta.download_url is None


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------


class TestIndiaCodeClassification:
    def test_classify_statute(self, ic_scraper):
        from src.acquisition.scrapers._india_code import _BrowseEntry

        url = "https://www.indiacode.nic.in/handle/123456789/1999?view_type=browse"
        ic_scraper._browse_metadata[url] = _BrowseEntry(
            handle="1999",
            detail_url=url,
            short_title="The Information Technology Act, 2000",
        )
        result = ic_scraper.classify_document(url, "<html></html>")
        assert result == DocumentType.STATUTE

    def test_classify_schedule(self, ic_scraper):
        from src.acquisition.scrapers._india_code import _BrowseEntry

        url = "https://www.indiacode.nic.in/handle/123456789/1505?view_type=browse"
        ic_scraper._browse_metadata[url] = _BrowseEntry(
            handle="1505",
            detail_url=url,
            short_title="FIRST SCHEDULE - The Indian Penal Code",
        )
        result = ic_scraper.classify_document(url, "<html></html>")
        assert result == DocumentType.SCHEDULE

    def test_classify_no_cache_defaults_statute(self, ic_scraper):
        result = ic_scraper.classify_document("https://example.com", "<html></html>")
        assert result == DocumentType.STATUTE


# ------------------------------------------------------------------
# Content format
# ------------------------------------------------------------------


class TestIndiaCodeContentFormat:
    def test_always_html(self, ic_scraper):
        assert ic_scraper.detect_content_format("any-url", "any-content") == ContentFormat.HTML


# ------------------------------------------------------------------
# Flags
# ------------------------------------------------------------------


class TestIndiaCodeFlags:
    def test_small_content_flagged(self, ic_scraper):
        flags = ic_scraper.detect_flags("<html>tiny</html>", ContentFormat.HTML)
        assert any(f.flag_type.value == "small_content" for f in flags)

    def test_missing_pdf_flagged(self, ic_scraper, sample_india_code_detail_no_pdf_html):
        flags = ic_scraper.detect_flags(sample_india_code_detail_no_pdf_html, ContentFormat.HTML)
        assert any(f.flag_type.value == "missing_metadata" for f in flags)

    def test_valid_detail_no_flags(self, ic_scraper, sample_india_code_detail_html):
        flags = ic_scraper.detect_flags(sample_india_code_detail_html, ContentFormat.HTML)
        flag_types = [f.flag_type.value for f in flags]
        assert "small_content" not in flag_types
        assert "missing_metadata" not in flag_types


# ------------------------------------------------------------------
# Document ID
# ------------------------------------------------------------------


class TestIndiaCodeDocId:
    def test_doc_id_from_handle_url(self, ic_scraper):
        assert (
            ic_scraper._doc_id_from_url(
                "https://www.indiacode.nic.in/handle/123456789/1999?view_type=browse"
            )
            == "ic_1999"
        )

    def test_doc_id_from_non_handle_url(self, ic_scraper):
        result = ic_scraper._doc_id_from_url("https://example.com/some/path/document")
        assert result == "document"


# ------------------------------------------------------------------
# Helper: _is_non_latin
# ------------------------------------------------------------------


class TestIsNonLatin:
    def test_english_text(self):
        assert _is_non_latin("The Information Technology Act, 2000") is False

    def test_hindi_text(self):
        assert _is_non_latin("सूचना प्रौद्योगिकी अधिनियम, 2000") is True

    def test_empty_string(self):
        assert _is_non_latin("") is False

    def test_numbers_only(self):
        assert _is_non_latin("12345") is False
