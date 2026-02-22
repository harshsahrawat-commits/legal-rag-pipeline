from __future__ import annotations

from datetime import UTC, datetime

from src.acquisition._models import CrawlRecord, CrawlState, SourceType
from src.acquisition.agents._change_detection import ChangeDetectionAgent


class TestChangeDetectionAgent:
    def test_all_new_urls(self):
        agent = ChangeDetectionAgent()
        state = CrawlState(source_type=SourceType.INDIAN_KANOON)
        urls = [
            "https://indiankanoon.org/doc/1/",
            "https://indiankanoon.org/doc/2/",
            "https://indiankanoon.org/doc/3/",
        ]

        results = agent.filter_urls(urls, state, SourceType.INDIAN_KANOON)
        assert len(results) == 3
        assert all(r.is_new for r in results)

    def test_filters_existing_urls(self):
        agent = ChangeDetectionAgent()
        state = CrawlState(
            source_type=SourceType.INDIAN_KANOON,
            records={
                "https://indiankanoon.org/doc/1/": CrawlRecord(
                    url="https://indiankanoon.org/doc/1/",
                    content_hash="hash1",
                    file_path="f1",
                    scraped_at=datetime(2024, 1, 1, tzinfo=UTC),
                ),
            },
        )
        urls = [
            "https://indiankanoon.org/doc/1/",  # existing
            "https://indiankanoon.org/doc/2/",  # new
        ]

        results = agent.filter_urls(urls, state, SourceType.INDIAN_KANOON)
        assert len(results) == 1
        assert results[0].url == "https://indiankanoon.org/doc/2/"
        assert results[0].is_new is True

    def test_empty_urls(self):
        agent = ChangeDetectionAgent()
        state = CrawlState(source_type=SourceType.INDIA_CODE)
        results = agent.filter_urls([], state, SourceType.INDIA_CODE)
        assert results == []

    def test_all_existing_returns_empty(self):
        agent = ChangeDetectionAgent()
        state = CrawlState(
            source_type=SourceType.INDIAN_KANOON,
            records={
                "url1": CrawlRecord(
                    url="url1",
                    content_hash="h1",
                    file_path="f1",
                    scraped_at=datetime(2024, 1, 1, tzinfo=UTC),
                ),
                "url2": CrawlRecord(
                    url="url2",
                    content_hash="h2",
                    file_path="f2",
                    scraped_at=datetime(2024, 1, 1, tzinfo=UTC),
                ),
            },
        )

        results = agent.filter_urls(["url1", "url2"], state, SourceType.INDIAN_KANOON)
        assert results == []
