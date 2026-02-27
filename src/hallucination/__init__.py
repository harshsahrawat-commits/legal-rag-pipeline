"""Phase 8: Hallucination Mitigation — post-generation verification."""

from src.hallucination._models import HallucinationConfig, VerifiedResponse
from src.hallucination.pipeline import HallucinationPipeline

__all__ = [
    "HallucinationConfig",
    "HallucinationPipeline",
    "VerifiedResponse",
]
