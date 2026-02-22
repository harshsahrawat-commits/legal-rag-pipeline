from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.acquisition._models import CrawlRecord, CrawlState, DocumentType, SourceType
from src.acquisition._state import CrawlStateStore

if TYPE_CHECKING:
    from pathlib import Path


class TestCrawlStateStore:
    def test_load_returns_empty_when_no_file(self, tmp_state_dir: Path):
        store = CrawlStateStore(tmp_state_dir)
        state = store.load(SourceType.INDIAN_KANOON)
        assert state.source_type == SourceType.INDIAN_KANOON
        assert state.records == {}
        assert state.last_run is None

    def test_save_and_load_roundtrip(self, tmp_state_dir: Path):
        store = CrawlStateStore(tmp_state_dir)
        state = CrawlState(source_type=SourceType.INDIAN_KANOON)
        record = CrawlRecord(
            url="https://indiankanoon.org/doc/1/",
            content_hash="abc123",
            file_path="data/raw/indian_kanoon/1.html",
            scraped_at=datetime(2024, 1, 1, tzinfo=UTC),
            document_type=DocumentType.JUDGMENT,
        )
        store.upsert_record(state, record)
        store.save(state)

        loaded = store.load(SourceType.INDIAN_KANOON)
        assert len(loaded.records) == 1
        assert loaded.records["https://indiankanoon.org/doc/1/"].content_hash == "abc123"
        assert loaded.last_run is not None

    def test_has_url(self, tmp_state_dir: Path):
        store = CrawlStateStore(tmp_state_dir)
        state = CrawlState(source_type=SourceType.INDIA_CODE)
        record = CrawlRecord(
            url="https://indiacode.nic.in/handle/123",
            content_hash="xyz",
            file_path="data/raw/india_code/123.html",
            scraped_at=datetime(2024, 6, 1, tzinfo=UTC),
        )
        store.upsert_record(state, record)
        store.save(state)

        assert store.has_url(SourceType.INDIA_CODE, "https://indiacode.nic.in/handle/123")
        assert not store.has_url(SourceType.INDIA_CODE, "https://indiacode.nic.in/handle/999")

    def test_get_record(self, tmp_state_dir: Path):
        store = CrawlStateStore(tmp_state_dir)
        state = CrawlState(source_type=SourceType.INDIAN_KANOON)
        record = CrawlRecord(
            url="https://indiankanoon.org/doc/5/",
            content_hash="hash5",
            file_path="data/raw/indian_kanoon/5.html",
            scraped_at=datetime(2024, 3, 1, tzinfo=UTC),
        )
        store.upsert_record(state, record)
        store.save(state)

        result = store.get_record(SourceType.INDIAN_KANOON, "https://indiankanoon.org/doc/5/")
        assert result is not None
        assert result.content_hash == "hash5"

        assert store.get_record(SourceType.INDIAN_KANOON, "nonexistent") is None

    def test_upsert_overwrites(self, tmp_state_dir: Path):
        store = CrawlStateStore(tmp_state_dir)
        state = CrawlState(source_type=SourceType.INDIAN_KANOON)
        r1 = CrawlRecord(
            url="https://indiankanoon.org/doc/1/",
            content_hash="old",
            file_path="f1",
            scraped_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        store.upsert_record(state, r1)

        r2 = CrawlRecord(
            url="https://indiankanoon.org/doc/1/",
            content_hash="new",
            file_path="f1",
            scraped_at=datetime(2024, 6, 1, tzinfo=UTC),
        )
        store.upsert_record(state, r2)
        assert state.records["https://indiankanoon.org/doc/1/"].content_hash == "new"

    def test_creates_state_dir_if_missing(self, tmp_path: Path):
        new_dir = tmp_path / "new_state"
        store = CrawlStateStore(new_dir)
        assert new_dir.exists()

        state = store.load(SourceType.INDIAN_KANOON)
        assert state.records == {}
