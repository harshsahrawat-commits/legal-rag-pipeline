from __future__ import annotations

from pathlib import Path

import pytest

from src.acquisition._models import (
    ContentFormat,
    DocumentType,
    PreliminaryMetadata,
    ScrapeConfig,
    SourceDefinition,
    SourceType,
)
from src.acquisition._state import CrawlStateStore
from src.acquisition.base_scraper import BaseScraper


class DummyScraper(BaseScraper):
    """Minimal concrete scraper for testing the base class."""

    def __init__(self, source_def, state_store, output_dir, urls=None, content="<html>test</html>"):
        super().__init__(source_def, state_store, output_dir)
        self._urls = urls or []
        self._content = content

    async def discover_urls(self, client):
        return self._urls

    def detect_content_format(self, url, content):
        return ContentFormat.HTML

    def classify_document(self, url, content):
        return DocumentType.JUDGMENT

    def extract_metadata(self, url, content):
        return PreliminaryMetadata(title="Test Document")


@pytest.fixture()
def source_def():
    return SourceDefinition(
        name="Test Source",
        source_type=SourceType.INDIAN_KANOON,
        base_url="https://example.com",
        rate_limit_requests_per_second=100.0,
        scrape_config=ScrapeConfig(max_documents=10),
    )


class TestBaseScraper:
    def test_source_type(self, source_def, tmp_state_dir, tmp_output_dir):
        store = CrawlStateStore(tmp_state_dir)
        scraper = DummyScraper(source_def, store, tmp_output_dir)
        assert scraper.source_type == SourceType.INDIAN_KANOON

    def test_output_dir_created(self, source_def, tmp_state_dir, tmp_output_dir):
        store = CrawlStateStore(tmp_state_dir)
        DummyScraper(source_def, store, tmp_output_dir)
        assert (tmp_output_dir / "indian_kanoon").is_dir()

    def test_doc_id_from_url(self, source_def, tmp_state_dir, tmp_output_dir):
        store = CrawlStateStore(tmp_state_dir)
        scraper = DummyScraper(source_def, store, tmp_output_dir)
        assert scraper._doc_id_from_url("https://example.com/doc/123/") == "123"
        assert scraper._doc_id_from_url("https://example.com/page") == "page"

    def test_save_document(self, source_def, tmp_state_dir, tmp_output_dir):
        from src.acquisition._models import ScrapedContent

        store = CrawlStateStore(tmp_state_dir)
        scraper = DummyScraper(source_def, store, tmp_output_dir)

        scraped = ScrapedContent(
            url="https://example.com/doc/42/",
            content="<html>Hello</html>",
            content_format=ContentFormat.HTML,
            content_hash="abcdef",
            status_code=200,
        )
        raw_doc = scraper._save_document(
            scraped,
            DocumentType.JUDGMENT,
            PreliminaryMetadata(title="Test"),
            [],
        )

        assert raw_doc.document_type == DocumentType.JUDGMENT
        assert Path(raw_doc.raw_content_path).exists()
        # Sidecar meta.json
        meta_path = Path(raw_doc.raw_content_path).with_suffix(".meta.json")
        assert meta_path.exists()
