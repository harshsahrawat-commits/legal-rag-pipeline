from __future__ import annotations

from src.chunking._models import ChunkingConfig, ChunkingResult, LegalChunk
from src.chunking.pipeline import ChunkingPipeline

__all__ = [
    "ChunkingConfig",
    "ChunkingPipeline",
    "ChunkingResult",
    "LegalChunk",
]


def run_chunking(**kwargs: object) -> None:
    """Convenience wrapper for CLI usage."""
    import asyncio

    pipeline = ChunkingPipeline()
    asyncio.run(pipeline.run(**kwargs))
