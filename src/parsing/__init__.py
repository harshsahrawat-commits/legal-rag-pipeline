from __future__ import annotations

from src.parsing._downloader import PdfDownloader
from src.parsing._models import ParsedDocument, ParsingResult
from src.parsing.pipeline import ParsingPipeline

__all__ = ["ParsedDocument", "ParsingPipeline", "ParsingResult", "PdfDownloader"]


def run_parsing(**kwargs: object) -> None:
    """Convenience wrapper for CLI usage."""
    import asyncio

    pipeline = ParsingPipeline()
    asyncio.run(pipeline.run(**kwargs))
