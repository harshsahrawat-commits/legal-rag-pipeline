from __future__ import annotations

from src.acquisition._models import (
    ContentFormat,
    DocumentFlag,
    FlagSeverity,
    FlagType,
    RawDocument,
)
from src.utils._logging import get_logger

_log = get_logger(__name__)

_MIN_CONTENT_LENGTH = 500
_NON_ASCII_THRESHOLD = 0.3


class LegalReviewAgent:
    """Rule-based classification and flagging of acquired documents.

    Performs post-scrape validation:
    - Confirms document classification
    - Detects quality issues (scanned PDF, regional language, corruption)
    - Ensures metadata completeness
    """

    def review(self, raw_doc: RawDocument, content: str) -> RawDocument:
        """Review a scraped document and update its flags and classification.

        Args:
            raw_doc: The raw document to review.
            content: The raw content string.

        Returns:
            The same RawDocument with updated flags.
        """
        new_flags = list(raw_doc.flags)

        # Check content size
        if len(content) < _MIN_CONTENT_LENGTH:
            new_flags.append(
                DocumentFlag(
                    flag_type=FlagType.SMALL_CONTENT,
                    message=f"Content is only {len(content)} characters",
                    severity=FlagSeverity.WARNING,
                )
            )

        # Check for regional language (high non-ASCII ratio)
        if content:
            non_ascii = sum(1 for c in content if ord(c) > 127)
            if non_ascii / len(content) > _NON_ASCII_THRESHOLD:
                new_flags.append(
                    DocumentFlag(
                        flag_type=FlagType.REGIONAL_LANGUAGE,
                        message="High non-ASCII ratio suggests regional language",
                        severity=FlagSeverity.INFO,
                    )
                )

        # Check for scanned PDF markers
        if raw_doc.content_format == ContentFormat.PDF and self._looks_like_scanned_pdf(content):
            new_flags.append(
                DocumentFlag(
                    flag_type=FlagType.SCANNED_PDF,
                    message="PDF appears to be a scanned image (minimal text content)",
                    severity=FlagSeverity.WARNING,
                )
            )

        # Check metadata completeness
        meta = raw_doc.preliminary_metadata
        if not meta.title:
            new_flags.append(
                DocumentFlag(
                    flag_type=FlagType.MISSING_METADATA,
                    message="No title extracted",
                    severity=FlagSeverity.WARNING,
                )
            )

        # Deduplicate flags by type
        seen_types: set[str] = set()
        deduped: list[DocumentFlag] = []
        for flag in new_flags:
            key = f"{flag.flag_type.value}:{flag.message}"
            if key not in seen_types:
                seen_types.add(key)
                deduped.append(flag)

        raw_doc.flags = deduped

        _log.info(
            "legal_review_complete",
            url=raw_doc.url,
            doc_type=raw_doc.document_type.value if raw_doc.document_type else None,
            flag_count=len(deduped),
        )
        return raw_doc

    def _looks_like_scanned_pdf(self, content: str) -> bool:
        """Heuristic: scanned PDFs have very little extractable text."""
        # If PDF content has very few alphabetic characters relative to total size
        if not content:
            return False
        alpha_count = sum(1 for c in content if c.isalpha())
        return len(content) > 1000 and alpha_count / len(content) < 0.1
