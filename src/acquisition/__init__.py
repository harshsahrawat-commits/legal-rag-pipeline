from __future__ import annotations

from src.acquisition.base_scraper import BaseScraper
from src.acquisition.pipeline import AcquisitionPipeline

__all__ = ["AcquisitionPipeline", "BaseScraper"]


def run_acquisition(**kwargs) -> None:
    """Convenience wrapper for CLI usage."""
    import asyncio

    pipeline = AcquisitionPipeline()
    asyncio.run(pipeline.run(**kwargs))
