from __future__ import annotations

from typing import TYPE_CHECKING

from aioresponses import aioresponses

from src.acquisition._models import (
    GlobalAcquisitionSettings,
    ScrapeConfig,
    SourceDefinition,
    SourceRegistry,
    SourceType,
)
from src.acquisition.pipeline import AcquisitionPipeline

if TYPE_CHECKING:
    from pathlib import Path

SEARCH_HTML = """
<html><body>
<div class="results_middle_col">
<div class="result"><div class="result_title"><a href="/doc/101/">State vs A on 1 Jan 2024</a></div></div>
<div class="result"><div class="result_title"><a href="/doc/102/">Section 302 in The Indian Penal Code</a></div></div>
</div>
</body></html>
"""

DOC_JUDGMENT_HTML = (
    """
<html>
<head><title>State vs A on 1 January, 2024</title></head>
<body>
<div class="doc_title">State vs A on 1 January, 2024</div>
<div class="doc_bench">Supreme Court of India</div>
<div class="judgments"><p>This is a judgment text with enough content to pass the minimum length check.
"""
    + "x" * 500
    + """
</p></div>
</body></html>
"""
)

DOC_STATUTE_HTML = (
    """
<html>
<head><title>Section 302 in The Indian Penal Code</title></head>
<body>
<div class="doc_title">Section 302 in The Indian Penal Code</div>
<div class="judgments"><p>302. Punishment for murder. Whoever commits murder shall be punished...</p>
"""
    + "x" * 500
    + """
</div>
</body></html>
"""
)

INDIA_CODE_BROWSE_HTML = """
<html><body>
<table class="table" summary="This table browses all dspace content">
<thead><tr><th>Enactment Date</th><th>Act Number</th><th>Short Title</th><th>View</th></tr></thead>
<tbody>
<tr>
<td>6-Oct-1860</td><td><em>45</em></td>
<td><strong>The Indian Penal Code, 1860</strong></td>
<td><a href="/handle/123456789/1505?view_type=browse">View...</a></td>
</tr>
</tbody>
</table>
</body></html>
"""

INDIA_CODE_BROWSE_EMPTY_HTML = """
<html><body>
<table class="table" summary="This table browses all dspace content">
<thead><tr><th>Enactment Date</th><th>Act Number</th><th>Short Title</th><th>View</th></tr></thead>
<tbody></tbody>
</table>
</body></html>
"""

INDIA_CODE_DETAIL_HTML = (
    """
<html><body>
<div class="item-summary-view-metadata">
<a href="/bitstream/123456789/1505/1/A1860-45.pdf">
<p>The Indian Penal Code, 1860</p>
</a>
</div>
<div class="sections"><p>Section 1. Title and extent of operation of the Code.</p>
"""
    + "x" * 500
    + """
</div>
</body></html>
"""
)


def _make_test_registry(tmp_path: Path) -> SourceRegistry:
    return SourceRegistry(
        settings=GlobalAcquisitionSettings(
            output_dir=tmp_path / "raw",
            state_dir=tmp_path / "state",
            concurrency=2,
        ),
        sources=[
            SourceDefinition(
                name="Indian Kanoon",
                source_type=SourceType.INDIAN_KANOON,
                base_url="https://indiankanoon.org",
                rate_limit_requests_per_second=100.0,
                request_timeout_seconds=10,
                max_retries=1,
                scrape_config=ScrapeConfig(
                    seed_queries=["IPC"],
                    max_pages_per_query=1,
                    max_documents=5,
                ),
            ),
            SourceDefinition(
                name="India Code",
                source_type=SourceType.INDIA_CODE,
                base_url="https://www.indiacode.nic.in",
                rate_limit_requests_per_second=100.0,
                request_timeout_seconds=10,
                max_retries=1,
                scrape_config=ScrapeConfig(
                    max_pages_per_query=1,
                    max_documents=5,
                ),
            ),
        ],
    )


class TestAcquisitionPipeline:
    async def test_pipeline_single_source(self, tmp_path: Path):
        registry = _make_test_registry(tmp_path)
        pipeline = AcquisitionPipeline(registry=registry)

        with aioresponses() as m:
            # Search page
            m.get(
                "https://indiankanoon.org/search/?formInput=IPC&pagenum=0",
                body=SEARCH_HTML,
            )
            # Doc pages
            m.get("https://indiankanoon.org/doc/101/", body=DOC_JUDGMENT_HTML)
            m.get("https://indiankanoon.org/doc/102/", body=DOC_STATUTE_HTML)

            results = await pipeline.run(source_name="Indian Kanoon")

        assert len(results) == 1
        result = results[0]
        assert result.source_type == SourceType.INDIAN_KANOON
        assert result.documents_downloaded == 2
        assert not result.errors

        # Check files exist
        output_dir = tmp_path / "raw" / "indian_kanoon"
        assert (output_dir / "101.html").exists()
        assert (output_dir / "101.meta.json").exists()
        assert (output_dir / "102.html").exists()
        assert (output_dir / "102.meta.json").exists()

        # Check state saved
        state_file = tmp_path / "state" / "indian_kanoon.json"
        assert state_file.exists()

    async def test_pipeline_idempotent(self, tmp_path: Path):
        """Second run with same content should download zero docs."""
        registry = _make_test_registry(tmp_path)
        pipeline = AcquisitionPipeline(registry=registry)

        with aioresponses() as m:
            m.get(
                "https://indiankanoon.org/search/?formInput=IPC&pagenum=0",
                body=SEARCH_HTML,
            )
            m.get("https://indiankanoon.org/doc/101/", body=DOC_JUDGMENT_HTML)
            m.get("https://indiankanoon.org/doc/102/", body=DOC_STATUTE_HTML)

            results1 = await pipeline.run(source_name="Indian Kanoon")

        assert results1[0].documents_downloaded == 2

        # Second run — same URLs, same content
        with aioresponses() as m:
            m.get(
                "https://indiankanoon.org/search/?formInput=IPC&pagenum=0",
                body=SEARCH_HTML,
            )
            m.get("https://indiankanoon.org/doc/101/", body=DOC_JUDGMENT_HTML)
            m.get("https://indiankanoon.org/doc/102/", body=DOC_STATUTE_HTML)

            results2 = await pipeline.run(source_name="Indian Kanoon")

        assert results2[0].documents_downloaded == 0

    async def test_pipeline_dry_run(self, tmp_path: Path):
        registry = _make_test_registry(tmp_path)
        pipeline = AcquisitionPipeline(registry=registry)

        with aioresponses() as m:
            m.get(
                "https://indiankanoon.org/search/?formInput=IPC&pagenum=0",
                body=SEARCH_HTML,
            )

            results = await pipeline.run(source_name="Indian Kanoon", dry_run=True)

        assert len(results) == 1
        assert results[0].documents_downloaded == 0
        # No files should be written
        output_dir = tmp_path / "raw" / "indian_kanoon"
        if output_dir.exists():
            assert len(list(output_dir.iterdir())) == 0

    async def test_pipeline_source_not_found(self, tmp_path: Path):
        registry = _make_test_registry(tmp_path)
        pipeline = AcquisitionPipeline(registry=registry)

        results = await pipeline.run(source_name="Nonexistent Source")
        assert results == []

    async def test_pipeline_multiple_sources(self, tmp_path: Path):
        import re

        registry = _make_test_registry(tmp_path)
        pipeline = AcquisitionPipeline(registry=registry)

        with aioresponses() as m:
            # Indian Kanoon
            m.get(
                "https://indiankanoon.org/search/?formInput=IPC&pagenum=0",
                body=SEARCH_HTML,
            )
            m.get("https://indiankanoon.org/doc/101/", body=DOC_JUDGMENT_HTML)
            m.get("https://indiankanoon.org/doc/102/", body=DOC_STATUTE_HTML)

            # India Code — browse listing (first page with 1 act, second empty)
            m.get(
                re.compile(
                    r"https://www\.indiacode\.nic\.in/handle/123456789/1362/browse\?.*offset=0.*"
                ),
                body=INDIA_CODE_BROWSE_HTML,
            )
            m.get(
                re.compile(
                    r"https://www\.indiacode\.nic\.in/handle/123456789/1362/browse\?.*offset=100.*"
                ),
                body=INDIA_CODE_BROWSE_EMPTY_HTML,
            )
            # India Code — detail page for the discovered act
            m.get(
                re.compile(r"https://www\.indiacode\.nic\.in/handle/123456789/1505\?.*"),
                body=INDIA_CODE_DETAIL_HTML,
            )

            results = await pipeline.run()

        assert len(results) == 2
        total = sum(r.documents_downloaded for r in results)
        assert total == 3

    async def test_pipeline_meta_json_valid(self, tmp_path: Path):
        """Check that .meta.json files are valid RawDocument JSON."""

        from src.acquisition._models import RawDocument

        registry = _make_test_registry(tmp_path)
        pipeline = AcquisitionPipeline(registry=registry)

        with aioresponses() as m:
            m.get(
                "https://indiankanoon.org/search/?formInput=IPC&pagenum=0",
                body=SEARCH_HTML,
            )
            m.get("https://indiankanoon.org/doc/101/", body=DOC_JUDGMENT_HTML)
            m.get("https://indiankanoon.org/doc/102/", body=DOC_STATUTE_HTML)

            await pipeline.run(source_name="Indian Kanoon")

        output_dir = tmp_path / "raw" / "indian_kanoon"
        for meta_file in output_dir.glob("*.meta.json"):
            raw = meta_file.read_text(encoding="utf-8")
            doc = RawDocument.model_validate_json(raw)
            assert doc.raw_content_path
            assert doc.content_format
            assert doc.source_type == SourceType.INDIAN_KANOON
