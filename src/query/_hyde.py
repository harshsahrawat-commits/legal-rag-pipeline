"""Selective HyDE — Hypothetical Document Embeddings for complex queries.

Generates a hypothetical answer via LLM for COMPLEX/ANALYTICAL queries,
then embeds it for vector search. Bridges the vocabulary gap between
how lawyers ask questions and how legal text is written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.query._models import HyDEResult
from src.utils._exceptions import LLMCallError, LLMNotAvailableError
from src.utils._llm_client import LLMMessage, get_llm_provider
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.query._models import QuerySettings
    from src.retrieval._models import QueryRoute
    from src.utils._llm_client import BaseLLMProvider

_log = get_logger(__name__)

_HYDE_PROMPT = (
    "You are an expert Indian lawyer. Given this legal research question, "
    "write a brief (2-3 sentence) hypothetical answer using proper Indian "
    "legal terminology, Act names, and section numbers. This will be used "
    "for retrieval, so include the specific legal terms that would appear "
    "in the relevant statutes or judgments.\n\n"
    "Question: {query}\n\n"
    "Hypothetical answer:"
)


class SelectiveHyDE:
    """Generates hypothetical document embeddings for complex queries.

    Only triggers for routes listed in ``settings.hyde_routes``
    (default: COMPLEX, ANALYTICAL). For other routes, returns a
    no-op result immediately.
    """

    def __init__(self, settings: QuerySettings) -> None:
        self._settings = settings
        self._provider: BaseLLMProvider | None = None

    @property
    def is_available(self) -> bool:
        """Check whether HyDE can be used (enabled + LLM provider available)."""
        if not self._settings.hyde_enabled:
            return False
        try:
            provider = get_llm_provider("hyde")
            return provider.is_available
        except LLMNotAvailableError:
            return False

    def _ensure_provider(self) -> None:
        """Lazy-initialize the LLM provider."""
        if self._provider is not None:
            return
        try:
            self._provider = get_llm_provider("hyde")
            _log.info("hyde_provider_initialized", provider=self._provider.provider_name)
        except LLMNotAvailableError:
            _log.warning("llm_provider_not_available")
        except Exception as exc:
            _log.warning("hyde_provider_init_failed", error=str(exc))

    def maybe_generate(
        self,
        query_text: str,
        route: QueryRoute,
        embedder: Any = None,
    ) -> HyDEResult:
        """Generate a hypothetical answer if the route warrants it.

        Args:
            query_text: The user's query.
            route: Classified query route from the router.
            embedder: Optional embedding model to embed the hypothetical.
                      Must have an ``embed_texts([str]) -> list[ndarray]`` method.

        Returns:
            HyDEResult with hypothetical_text and hyde_embedding if generated,
            or a no-op result if skipped/failed.
        """
        # Only generate for configured routes
        if route.value not in self._settings.hyde_routes:
            return HyDEResult(generated=False)

        if not self.is_available:
            _log.debug("hyde_skipped_not_available")
            return HyDEResult(generated=False)

        # Generate hypothetical answer
        hypothetical = self._generate_hypothetical(query_text)
        if hypothetical is None:
            return HyDEResult(generated=False)

        # Embed the hypothetical text
        hyde_embedding = self._embed_hypothetical(hypothetical, embedder)

        return HyDEResult(
            hypothetical_text=hypothetical,
            hyde_embedding=hyde_embedding,
            generated=True,
        )

    def _generate_hypothetical(self, query_text: str) -> str | None:
        """Call LLM to generate a hypothetical answer."""
        self._ensure_provider()
        if self._provider is None:
            _log.warning("hyde_no_provider")
            return None

        prompt = _HYDE_PROMPT.format(query=query_text)

        try:
            response = self._provider.complete(
                [LLMMessage(role="user", content=prompt)],
                max_tokens=self._settings.hyde_max_tokens,
            )
            text = response.text.strip()
            if text:
                _log.info(
                    "hyde_generated",
                    query_len=len(query_text),
                    hypothesis_len=len(text),
                )
                return text
            _log.warning("hyde_empty_response")
            return None
        except LLMCallError as exc:
            _log.warning("hyde_generation_failed", error=str(exc))
            return None
        except Exception as exc:
            _log.warning("hyde_generation_failed", error=str(exc))
            return None

    @staticmethod
    def _embed_hypothetical(
        hypothetical: str,
        embedder: Any,
    ) -> list[float] | None:
        """Embed the hypothetical text using the provided embedder."""
        if embedder is None:
            return None

        try:
            embeddings = embedder.embed_texts([hypothetical])
            if embeddings and len(embeddings) > 0:
                return embeddings[0].tolist()
        except Exception as exc:
            _log.warning("hyde_embedding_failed", error=str(exc))

        return None
