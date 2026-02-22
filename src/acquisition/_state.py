from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from src.acquisition._models import CrawlRecord, CrawlState, SourceType
from src.utils._logging import get_logger

_log = get_logger(__name__)


class CrawlStateStore:
    """File-based crawl state persistence.

    State is stored as one JSON file per source in the state directory.
    Writes are atomic via tmp-file + rename.
    """

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, source_type: SourceType) -> Path:
        return self._state_dir / f"{source_type.value}.json"

    def load(self, source_type: SourceType) -> CrawlState:
        """Load crawl state for a source. Returns empty state if no file exists."""
        path = self._path_for(source_type)
        if not path.exists():
            _log.info("no_existing_state", source=source_type.value)
            return CrawlState(source_type=source_type)

        raw = path.read_text(encoding="utf-8")
        state = CrawlState.model_validate_json(raw)
        _log.info(
            "state_loaded",
            source=source_type.value,
            record_count=len(state.records),
        )
        return state

    def save(self, state: CrawlState) -> None:
        """Atomically save crawl state for a source."""
        state.last_run = datetime.now(UTC)
        path = self._path_for(state.source_type)

        # Write to temp file first, then rename for atomicity
        fd, tmp_path = tempfile.mkstemp(dir=str(self._state_dir), suffix=".tmp", prefix="state_")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))
            Path(tmp_path).replace(path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        _log.info(
            "state_saved",
            source=state.source_type.value,
            record_count=len(state.records),
        )

    def has_url(self, source_type: SourceType, url: str) -> bool:
        """Check if a URL has been previously crawled."""
        state = self.load(source_type)
        return url in state.records

    def get_record(self, source_type: SourceType, url: str) -> CrawlRecord | None:
        """Get the crawl record for a specific URL, or None."""
        state = self.load(source_type)
        return state.records.get(url)

    def upsert_record(self, state: CrawlState, record: CrawlRecord) -> None:
        """Add or update a crawl record in the state (in-memory). Call save() to persist."""
        state.records[record.url] = record
