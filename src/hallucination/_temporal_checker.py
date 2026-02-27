"""Layer 2: Temporal Consistency Checking.

Verifies that legal provisions referenced in the response are
in force as of the reference date. Special handling for the
IPC→BNS, CrPC→BNSS, Evidence Act→BSA transition (July 1, 2024).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from src.hallucination._citation_extractor import extract_citations
from src.hallucination._models import CitationType, TemporalWarning
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.hallucination._models import HallucinationSettings
    from src.knowledge_graph._queries import QueryBuilder

_log = get_logger(__name__)

# Hardcoded IPC→BNS / CrPC→BNSS / Evidence Act→BSA transition
_REPEALED_ACTS: dict[str, dict[str, str]] = {
    "Indian Penal Code": {
        "replacement_act": "Bharatiya Nyaya Sanhita",
        "repealer": "Bharatiya Nyaya Sanhita, 2023",
    },
    "Code of Criminal Procedure": {
        "replacement_act": "Bharatiya Nagarik Suraksha Sanhita",
        "repealer": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    },
    "Indian Evidence Act": {
        "replacement_act": "Bharatiya Sakshya Adhiniyam",
        "repealer": "Bharatiya Sakshya Adhiniyam, 2023",
    },
}


def get_repealed_acts() -> dict[str, dict[str, str]]:
    """Public accessor for the repealed acts mapping."""
    return dict(_REPEALED_ACTS)


class TemporalChecker:
    """Check whether referenced legal provisions are temporally valid."""

    def __init__(
        self,
        settings: HallucinationSettings,
        query_builder: QueryBuilder | None = None,
    ) -> None:
        self._settings = settings
        self._qb = query_builder
        self._repeal_date = date.fromisoformat(settings.ipc_repeal_date)

    async def check_response(
        self,
        response_text: str,
        reference_date: date | None = None,
    ) -> list[TemporalWarning]:
        """Check all section references in the response for temporal issues."""
        ref_date = reference_date or date.today()
        citations = extract_citations(response_text)
        section_refs = [c for c in citations if c.citation_type == CitationType.SECTION_REF]

        if not section_refs:
            return []

        warnings: list[TemporalWarning] = []
        seen: set[tuple[str, str]] = set()

        for citation in section_refs:
            if not citation.section or not citation.act:
                continue
            key = (citation.section, citation.act)
            if key in seen:
                continue
            seen.add(key)

            # Check hardcoded repeals first (no KG needed)
            hardcoded = self._check_hardcoded_repeal(citation.section, citation.act, ref_date)
            if hardcoded:
                warnings.append(hardcoded)
                continue

            # Check KG for temporal status
            kg_warning = await self._check_kg_temporal(citation.section, citation.act, ref_date)
            if kg_warning:
                warnings.append(kg_warning)

        return warnings

    def _check_hardcoded_repeal(
        self, section: str, act: str, ref_date: date
    ) -> TemporalWarning | None:
        """Check if the act is one of the three repealed codes (post July 2024)."""
        repeal_info = _REPEALED_ACTS.get(act)
        if repeal_info is None:
            return None

        # Only warn if reference date is on or after the repeal date
        if ref_date < self._repeal_date:
            return None

        replacement_act = repeal_info["replacement_act"]
        repealer = repeal_info["repealer"]

        return TemporalWarning(
            section=section,
            act=act,
            warning_text=(
                f"Section {section} of {act} was repealed on {self._repeal_date}. "
                f"The replacement law is {replacement_act}."
            ),
            repealed_by=repealer,
            replacement_act=replacement_act,
            reference_date=ref_date,
        )

    async def _check_kg_temporal(
        self, section: str, act: str, ref_date: date
    ) -> TemporalWarning | None:
        """Query KG for temporal status of a section."""
        if self._qb is None:
            return None

        try:
            status = await self._qb.temporal_status(section=section, act=act, ref_date=ref_date)
        except Exception as exc:
            _log.warning(
                "temporal_kg_query_failed",
                section=section,
                act=act,
                error=str(exc),
            )
            return None

        if not status.get("found"):
            return None

        # Check if section is not in force
        if status.get("is_in_force") is False:
            # Try to find replacement
            replacement = await self._find_replacement(act, section)
            return TemporalWarning(
                section=section,
                act=act,
                warning_text=(f"Section {section} of {act} is no longer in force."),
                repealed_by=status.get("act_status"),
                replacement_act=replacement.get("replacement_act") if replacement else None,
                replacement_section=replacement.get("replacement_section") if replacement else None,
                reference_date=ref_date,
            )

        # Check act-level repeal
        act_status = status.get("act_status", "")
        if act_status == "repealed":
            replacement = await self._find_replacement(act, section)
            return TemporalWarning(
                section=section,
                act=act,
                warning_text=f"The {act} has been repealed.",
                repealed_by=act_status,
                replacement_act=replacement.get("replacement_act") if replacement else None,
                replacement_section=replacement.get("replacement_section") if replacement else None,
                reference_date=ref_date,
            )

        return None

    async def _find_replacement(self, act: str, section: str) -> dict[str, str | None]:
        """Try to find the replacement act/section via KG."""
        if self._qb is None:
            return {}
        try:
            result = await self._qb.find_replacement(act, section)
            return result or {}
        except Exception as exc:
            _log.warning(
                "find_replacement_failed",
                act=act,
                section=section,
                error=str(exc),
            )
            return {}
