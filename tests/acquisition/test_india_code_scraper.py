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
from src.acquisition.scrapers._india_code import IndiaCodeScraper


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
            seed_act_ids=["45_of_1860", "2_of_1974"],
            max_documents=10,
        ),
    )


@pytest.fixture()
def ic_scraper(ic_source_def, tmp_state_dir, tmp_output_dir):
    store = CrawlStateStore(tmp_state_dir)
    return IndiaCodeScraper(ic_source_def, store, tmp_output_dir)


class TestIndiaCodeClassification:
    def test_classify_statute(self, ic_scraper, sample_india_code_act_html):
        result = ic_scraper.classify_document(
            "https://www.indiacode.nic.in/handle/123456789/1362",
            sample_india_code_act_html,
        )
        assert result == DocumentType.STATUTE

    def test_classify_schedule(self, ic_scraper):
        html = """
        <html><body>
        <div class="actTitle">FIRST SCHEDULE - The Indian Penal Code</div>
        </body></html>
        """
        result = ic_scraper.classify_document("url", html)
        assert result == DocumentType.SCHEDULE


class TestIndiaCodeMetadata:
    def test_extract_metadata(self, ic_scraper, sample_india_code_act_html):
        meta = ic_scraper.extract_metadata(
            "https://www.indiacode.nic.in/handle/123456789/1362",
            sample_india_code_act_html,
        )
        assert meta.title == "THE INDIAN PENAL CODE, 1860"
        assert meta.act_name == "THE INDIAN PENAL CODE, 1860"
        assert meta.act_number == "Act No. 45 of 1860"
        assert meta.year == 1860
        assert meta.date == "6th October, 1860"


class TestIndiaCodeDiscovery:
    async def test_discover_urls(self, ic_scraper):
        from src.acquisition._http_client import HttpClient
        from src.acquisition._rate_limiter import AsyncRateLimiter

        limiter = AsyncRateLimiter(100.0)
        async with HttpClient(rate_limiter=limiter) as client:
            urls = await ic_scraper.discover_urls(client)

        assert len(urls) == 2
        assert "45_of_1860" in urls[0]
        assert "2_of_1974" in urls[1]


class TestIndiaCodeContentFormat:
    def test_html_format(self, ic_scraper):
        assert ic_scraper.detect_content_format("url.html", "<html>") == ContentFormat.HTML

    def test_pdf_by_url(self, ic_scraper):
        assert ic_scraper.detect_content_format("url.pdf", "content") == ContentFormat.PDF

    def test_pdf_by_content(self, ic_scraper):
        assert ic_scraper.detect_content_format("url", "%PDF-1.4") == ContentFormat.PDF


class TestIndiaCodeFlags:
    def test_small_content_flagged(self, ic_scraper):
        flags = ic_scraper.detect_flags("<html>tiny</html>", ContentFormat.HTML)
        assert any(f.flag_type.value == "small_content" for f in flags)

    def test_corrupt_pdf_flagged(self, ic_scraper):
        flags = ic_scraper.detect_flags("<html>not a pdf</html>", ContentFormat.PDF)
        assert any(f.flag_type.value == "corrupt_content" for f in flags)


class TestIndiaCodeDocId:
    def test_doc_id_from_url(self, ic_scraper):
        assert (
            ic_scraper._doc_id_from_url("https://www.indiacode.nic.in/handle/123456789/45_of_1860")
            == "45_of_1860"
        )
