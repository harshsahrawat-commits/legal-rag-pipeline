"""Layer 4: GenGround Verification.

SIMPLE route: single-pass LLM audit (1 call).
STANDARD+: full per-claim decomposition, re-retrieval, and alignment checking.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.hallucination._exceptions import GenGroundError, GenGroundNotAvailableError
from src.hallucination._models import (
    ClaimVerdict,
    ClaimVerdictType,
    ExtractedClaim,
)
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.hallucination._models import HallucinationSettings
    from src.retrieval._engine import RetrievalEngine
    from src.retrieval._models import ExpandedContext

_log = get_logger(__name__)

# --- Prompt templates ---

_SIMPLE_AUDIT_PROMPT = """You are a legal accuracy auditor. Given a response and source chunks, identify any claims that are NOT supported by the sources.

Response to audit:
{response}

Source chunks:
{sources}

Return a JSON object with:
- "issues": list of strings describing unsupported or inaccurate claims (empty list if all supported)
- "verdict": "supported" if all claims are grounded, "partially_supported" if some issues, "unsupported" if major issues

Return ONLY valid JSON, no other text."""

_CLAIM_EXTRACTION_PROMPT = """Extract atomic factual claims from this legal response. Each claim should be a single, verifiable statement.

Response:
{response}

Return a JSON array of objects, each with:
- "claim_id": integer starting from 1
- "text": the atomic factual claim
- "source_span": the relevant phrase from the original response

Return ONLY valid JSON array, no other text."""

_CLAIM_ALIGNMENT_PROMPT = """Determine if this claim is supported by the evidence chunks.

Claim: {claim}

Evidence chunks:
{evidence}

Return a JSON object with:
- "verdict": one of "supported", "unsupported", "partially_supported"
- "confidence": float 0-1
- "reasoning": brief explanation
- "issues": list of strings (empty if supported)

Return ONLY valid JSON, no other text."""


class GenGroundRefiner:
    """GenGround verification: audit and correct LLM responses."""

    def __init__(
        self,
        settings: HallucinationSettings,
        retrieval_engine: RetrievalEngine | None = None,
    ) -> None:
        self._settings = settings
        self._engine = retrieval_engine
        self._client: Any = None
        self._llm_calls = 0

    @property
    def llm_calls(self) -> int:
        """Number of LLM calls made during the last verify() call."""
        return self._llm_calls

    async def verify(
        self,
        response_text: str,
        chunks: list[ExpandedContext],
        is_simple: bool = False,
    ) -> tuple[str, list[ClaimVerdict]]:
        """Verify the response via GenGround.

        Args:
            response_text: The LLM response to verify.
            chunks: Source chunks from retrieval.
            is_simple: If True, use single-pass audit instead of per-claim.

        Returns:
            Tuple of (modified_response, list of ClaimVerdict).
        """
        self._llm_calls = 0

        if not self._settings.genground_enabled:
            return response_text, []

        if is_simple:
            return await self._simple_audit(response_text, chunks)
        return await self._full_verify(response_text, chunks)

    async def _simple_audit(
        self,
        response_text: str,
        chunks: list[ExpandedContext],
    ) -> tuple[str, list[ClaimVerdict]]:
        """Single-pass LLM audit — 1 call."""
        sources_text = "\n\n".join(f"[Chunk {i + 1}]: {c.chunk_text}" for i, c in enumerate(chunks))
        prompt = _SIMPLE_AUDIT_PROMPT.format(response=response_text, sources=sources_text)

        raw = await self._llm_call(prompt)
        self._llm_calls += 1

        parsed = _parse_json(raw)
        issues = parsed.get("issues", [])
        verdict_str = parsed.get("verdict", "supported")

        try:
            verdict_type = ClaimVerdictType(verdict_str)
        except ValueError:
            verdict_type = ClaimVerdictType.PARTIALLY_SUPPORTED

        claim_verdict = ClaimVerdict(
            claim=ExtractedClaim(claim_id=1, text="[entire response]"),
            verdict=verdict_type,
            confidence=1.0 if verdict_type == ClaimVerdictType.SUPPORTED else 0.5,
            issues=issues if isinstance(issues, list) else [],
            reasoning=f"Simple audit: {verdict_str}",
        )
        return response_text, [claim_verdict]

    async def _full_verify(
        self,
        response_text: str,
        chunks: list[ExpandedContext],
    ) -> tuple[str, list[ClaimVerdict]]:
        """Full per-claim verification — multiple LLM calls."""
        # Step 1: Extract claims
        claims = await self._extract_claims(response_text)
        if not claims:
            return response_text, []

        # Step 2+3: Per-claim re-retrieval + alignment
        verdicts: list[ClaimVerdict] = []
        for claim in claims[: self._settings.genground_max_claims]:
            verdict = await self._verify_claim(claim, chunks)
            verdicts.append(verdict)

        # Step 4: Reconstruct response
        modified = self._reconstruct_response(response_text, verdicts)

        return modified, verdicts

    async def _extract_claims(self, response_text: str) -> list[ExtractedClaim]:
        """Extract atomic claims from the response (1 LLM call)."""
        prompt = _CLAIM_EXTRACTION_PROMPT.format(response=response_text)
        raw = await self._llm_call(prompt)
        self._llm_calls += 1

        parsed = _parse_json(raw)

        claims: list[ExtractedClaim] = []
        items = parsed if isinstance(parsed, list) else parsed.get("claims", [])
        for item in items:
            if isinstance(item, dict) and "text" in item:
                claims.append(
                    ExtractedClaim(
                        claim_id=item.get("claim_id", len(claims) + 1),
                        text=item["text"],
                        source_span=item.get("source_span"),
                    )
                )
        return claims

    async def _verify_claim(
        self,
        claim: ExtractedClaim,
        initial_chunks: list[ExpandedContext],
    ) -> ClaimVerdict:
        """Re-retrieve and verify a single claim (1 LLM call + optional re-retrieval)."""
        # Re-retrieve for this specific claim
        evidence_chunk_ids: list[str] = [c.chunk_id for c in initial_chunks]

        if self._engine is not None:
            try:
                re_retrieved = await self._engine.hybrid_search(
                    claim.text,
                    top_k=self._settings.genground_re_retrieval_top_k,
                )
                if re_retrieved:
                    evidence_chunk_ids = [c.chunk_id for c in re_retrieved]
                    # Build evidence text from re-retrieved chunks
                    evidence_text = "\n\n".join(f"[{c.chunk_id}]: {c.text}" for c in re_retrieved)
                else:
                    evidence_text = "\n\n".join(
                        f"[{c.chunk_id}]: {c.chunk_text}" for c in initial_chunks
                    )
            except Exception as exc:
                _log.warning("genground_re_retrieval_failed", error=str(exc))
                evidence_text = "\n\n".join(
                    f"[{c.chunk_id}]: {c.chunk_text}" for c in initial_chunks
                )
        else:
            evidence_text = "\n\n".join(f"[{c.chunk_id}]: {c.chunk_text}" for c in initial_chunks)

        # LLM alignment check
        prompt = _CLAIM_ALIGNMENT_PROMPT.format(claim=claim.text, evidence=evidence_text)
        raw = await self._llm_call(prompt)
        self._llm_calls += 1

        parsed = _parse_json(raw)

        verdict_str = parsed.get("verdict", "unsupported")
        try:
            verdict_type = ClaimVerdictType(verdict_str)
        except ValueError:
            verdict_type = ClaimVerdictType.PARTIALLY_SUPPORTED

        return ClaimVerdict(
            claim=claim,
            verdict=verdict_type,
            confidence=float(parsed.get("confidence", 0.5)),
            evidence_chunk_ids=evidence_chunk_ids,
            issues=parsed.get("issues", []),
            reasoning=parsed.get("reasoning", ""),
        )

    def _reconstruct_response(self, response_text: str, verdicts: list[ClaimVerdict]) -> str:
        """Add caveats for unsupported/partially-supported claims."""
        unsupported = [v for v in verdicts if v.verdict == ClaimVerdictType.UNSUPPORTED]
        partial = [v for v in verdicts if v.verdict == ClaimVerdictType.PARTIALLY_SUPPORTED]

        if not unsupported and not partial:
            return response_text

        caveats: list[str] = []
        if unsupported:
            caveats.append(
                f"\n\n[Note: {len(unsupported)} claim(s) could not be verified "
                "against available sources and may be inaccurate.]"
            )
        if partial:
            caveats.append(
                f"\n\n[Note: {len(partial)} claim(s) are only partially "
                "supported by available sources.]"
            )

        return response_text + "".join(caveats)

    async def _llm_call(self, prompt: str) -> str:
        """Make an LLM call via AsyncAnthropic."""
        self._ensure_client()
        try:
            response = await self._client.messages.create(
                model=self._settings.llm_model,
                max_tokens=self._settings.llm_max_tokens,
                temperature=self._settings.llm_temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract text from response
            content = response.content
            if content and len(content) > 0:
                return content[0].text
            return ""
        except Exception as exc:
            msg = f"LLM call failed: {exc}"
            raise GenGroundError(msg) from exc

    def _ensure_client(self) -> None:
        """Lazy-initialize the Anthropic client."""
        if self._client is not None:
            return
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            msg = "anthropic package required. Install with: pip install anthropic"
            raise GenGroundNotAvailableError(msg) from exc
        self._client = AsyncAnthropic()


def _parse_json(raw: str) -> Any:
    """Parse JSON from LLM response, with fallback for markdown code blocks."""
    text = raw.strip()
    # Strip markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        _log.warning("json_parse_failed", raw_preview=text[:200])
        return {}
