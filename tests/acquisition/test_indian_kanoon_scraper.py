from __future__ import annotations

import pytest
from aioresponses import aioresponses

from src.acquisition._models import (
    ContentFormat,
    DocumentType,
    ScrapeConfig,
    SourceDefinition,
    SourceType,
)
from src.acquisition._state import CrawlStateStore
from src.acquisition.scrapers._indian_kanoon import IndianKanoonScraper


@pytest.fixture()
def ik_source_def():
    return SourceDefinition(
        name="Indian Kanoon",
        source_type=SourceType.INDIAN_KANOON,
        base_url="https://indiankanoon.org",
        rate_limit_requests_per_second=100.0,  # Fast for tests
        request_timeout_seconds=10,
        max_retries=1,
        scrape_config=ScrapeConfig(
            seed_queries=["Indian Penal Code"],
            max_pages_per_query=1,
            max_documents=5,
        ),
    )


@pytest.fixture()
def ik_scraper(ik_source_def, tmp_state_dir, tmp_output_dir):
    store = CrawlStateStore(tmp_state_dir)
    return IndianKanoonScraper(ik_source_def, store, tmp_output_dir)


class TestIndianKanoonClassification:
    def test_classify_judgment(self, ik_scraper, sample_judgment_html):
        result = ik_scraper.classify_document(
            "https://indiankanoon.org/doc/123/", sample_judgment_html
        )
        assert result == DocumentType.JUDGMENT

    def test_classify_statute(self, ik_scraper, sample_statute_html):
        result = ik_scraper.classify_document(
            "https://indiankanoon.org/doc/456/", sample_statute_html
        )
        assert result == DocumentType.STATUTE

    def test_classify_unknown_returns_none(self, ik_scraper):
        html = "<html><head><title>Some Random Page</title></head><body></body></html>"
        result = ik_scraper.classify_document("https://indiankanoon.org/doc/999/", html)
        assert result is None


class TestIndianKanoonMetadata:
    def test_extract_judgment_metadata(self, ik_scraper, sample_judgment_html):
        meta = ik_scraper.extract_metadata(
            "https://indiankanoon.org/doc/123/", sample_judgment_html
        )
        assert meta.title is not None
        assert "Maharashtra" in meta.title
        assert meta.court == "Supreme Court of India"
        assert meta.case_citation == "AIR 2024 SC 1500"

    def test_extract_statute_metadata(self, ik_scraper, sample_statute_html):
        meta = ik_scraper.extract_metadata("https://indiankanoon.org/doc/456/", sample_statute_html)
        assert meta.title is not None
        assert "Section 302" in meta.title
        assert meta.act_name is not None
        assert "Indian Penal Code" in meta.act_name


class TestIndianKanoonDiscovery:
    def test_extract_doc_urls(self, ik_scraper, sample_search_html):
        urls = ik_scraper._extract_doc_urls("https://indiankanoon.org", sample_search_html)
        assert len(urls) == 3
        assert all("/doc/" in u for u in urls)
        assert all(u.endswith("/") for u in urls)

    async def test_discover_urls_with_mock(self, ik_scraper, sample_search_html):
        with aioresponses() as m:
            m.get(
                "https://indiankanoon.org/search/?formInput=Indian+Penal+Code&pagenum=0",
                body=sample_search_html,
            )

            from src.acquisition._http_client import HttpClient
            from src.acquisition._rate_limiter import AsyncRateLimiter

            limiter = AsyncRateLimiter(100.0)
            async with HttpClient(rate_limiter=limiter) as client:
                urls = await ik_scraper.discover_urls(client)

            assert len(urls) == 3


class TestIndianKanoonFlags:
    def test_small_content_flagged(self, ik_scraper):
        flags = ik_scraper.detect_flags("<html>tiny</html>", ContentFormat.HTML)
        assert any(f.flag_type.value == "small_content" for f in flags)

    def test_normal_content_no_flag(self, ik_scraper):
        content = "<html>" + "x" * 1000 + "</html>"
        flags = ik_scraper.detect_flags(content, ContentFormat.HTML)
        assert not any(f.flag_type.value == "small_content" for f in flags)


class TestIndianKanoonDocId:
    def test_doc_id_from_url(self, ik_scraper):
        assert ik_scraper._doc_id_from_url("https://indiankanoon.org/doc/12345/") == "12345"
        assert ik_scraper._doc_id_from_url("https://indiankanoon.org/doc/999") == "999"


class TestIndianKanoonContentFormat:
    def test_always_html(self, ik_scraper):
        assert ik_scraper.detect_content_format("any_url", "any_content") == ContentFormat.HTML
