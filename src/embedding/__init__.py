from __future__ import annotations

from src.embedding._models import EmbeddingConfig, EmbeddingResult
from src.embedding.pipeline import EmbeddingPipeline

__all__ = [
    "EmbeddingConfig",
    "EmbeddingPipeline",
    "EmbeddingResult",
]


def run_embedding(**kwargs: object) -> None:
    """Convenience wrapper for CLI usage."""
    import asyncio

    pipeline = EmbeddingPipeline()
    asyncio.run(pipeline.run(**kwargs))
