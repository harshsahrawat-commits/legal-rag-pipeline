from __future__ import annotations

from src.enrichment._models import EnrichmentConfig, EnrichmentResult
from src.enrichment.pipeline import EnrichmentPipeline

__all__ = [
    "EnrichmentConfig",
    "EnrichmentPipeline",
    "EnrichmentResult",
]


def run_enrichment(**kwargs: object) -> None:
    """Convenience wrapper for CLI usage."""
    import asyncio

    pipeline = EnrichmentPipeline()
    asyncio.run(pipeline.run(**kwargs))
