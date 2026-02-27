"""Adaptive Query Router — rule-based classification for retrieval routing.

Classifies legal queries into SIMPLE / STANDARD / COMPLEX / ANALYTICAL
routes based on regex patterns and heuristics.
"""

from __future__ import annotations

import re

from src.query._models import QuerySettings, RouterResult
from src.retrieval._models import QueryRoute
from src.retrieval._searchers import extract_section_references
from src.utils._logging import get_logger

_log = get_logger(__name__)

# --- Act reference extraction ---

_ACT_ALIASES: dict[str, str] = {
    "ipc": "Indian Penal Code",
    "crpc": "Code of Criminal Procedure",
    "cpc": "Code of Civil Procedure",
    "bns": "Bharatiya Nyaya Sanhita",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita",
    "bsa": "Bharatiya Sakshya Adhiniyam",
    "ni act": "Negotiable Instruments Act",
    "it act": "Income Tax Act",
    "gst": "Goods and Services Tax Act",
    "sarfaesi": "SARFAESI Act",
    "rera": "Real Estate Regulation Act",
    "posh": "Prevention of Sexual Harassment Act",
    "dv act": "Domestic Violence Act",
    "mv act": "Motor Vehicles Act",
    "pocso": "Protection of Children from Sexual Offences Act",
}

# Build alias pattern: match multi-word aliases first (longest match), then single-word
_ALIAS_PATTERN: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ACT_ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Full act name pattern: matches "X Act[, YYYY]" and "X Code[, YYYY]"
_FULL_ACT_PATTERN: re.Pattern[str] = re.compile(
    r"(?:the\s+)?([\w][\w\s]*?(?:Act|Code|Sanhita|Adhiniyam)(?:,?\s+\d{4})?)\b",
    re.IGNORECASE,
)


def extract_act_references(text: str) -> list[str]:
    """Extract act name references from query text.

    Matches both common abbreviations (IPC, CrPC, NI Act, etc.) and
    full act names like "Indian Penal Code", "Companies Act, 2013".

    Returns:
        Deduplicated list of act names found (abbreviations resolved to full names).
    """
    acts: list[str] = []
    seen_lower: set[str] = set()

    # 1. Check abbreviations first
    for match in _ALIAS_PATTERN.finditer(text):
        alias = match.group(1).lower()
        full_name = _ACT_ALIASES.get(alias, alias)
        if full_name.lower() not in seen_lower:
            seen_lower.add(full_name.lower())
            acts.append(full_name)

    # 2. Check full act name pattern
    for match in _FULL_ACT_PATTERN.finditer(text):
        act_name = match.group(1).strip()
        if act_name.lower() not in seen_lower:
            seen_lower.add(act_name.lower())
            acts.append(act_name)

    return acts


# --- Adaptive Query Router ---


class AdaptiveQueryRouter:
    """Rule-based classifier routing queries to retrieval paths.

    Classification priority:
    1. SIMPLE — direct section/definition lookups (confidence 1.0)
    2. ANALYTICAL — comparative, historical, exhaustive queries (0.9)
    3. COMPLEX — multi-act, multi-section, multi-hop queries (0.8)
    4. STANDARD — everything else (0.5)
    """

    def __init__(self, settings: QuerySettings) -> None:
        self._settings = settings

        # --- SIMPLE patterns (compiled once) ---
        self._simple_patterns: list[tuple[re.Pattern[str], str]] = [
            (
                re.compile(r"what\s+does\s+section\s+\d+", re.IGNORECASE),
                "pattern:what_does_section",
            ),
            (
                re.compile(r"what\s+is\s+section\s+\d+", re.IGNORECASE),
                "pattern:what_is_section",
            ),
            (
                re.compile(r"what\s+is\s+article\s+\d+", re.IGNORECASE),
                "pattern:what_is_article",
            ),
            (
                re.compile(r"what\s+does\s+article\s+\d+", re.IGNORECASE),
                "pattern:what_does_article",
            ),
            (
                re.compile(r"define\s+['\"]?\w+['\"]?", re.IGNORECASE),
                "pattern:define",
            ),
            (
                re.compile(r"bare\s+text\s+of", re.IGNORECASE),
                "pattern:bare_text_of",
            ),
            (
                re.compile(r"text\s+of\s+section", re.IGNORECASE),
                "pattern:text_of_section",
            ),
            (
                re.compile(r"read\s+section", re.IGNORECASE),
                "pattern:read_section",
            ),
            (
                re.compile(r"show\s+me\s+section", re.IGNORECASE),
                "pattern:show_me_section",
            ),
            (
                re.compile(r"meaning\s+of\s+section", re.IGNORECASE),
                "pattern:meaning_of_section",
            ),
        ]

        # --- ANALYTICAL signal patterns ---
        self._analytical_patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\bcompare\b", re.IGNORECASE), "signal:compare"),
            (re.compile(r"\bcontrast\b", re.IGNORECASE), "signal:contrast"),
            (re.compile(r"\bevolution\b", re.IGNORECASE), "signal:evolution"),
            (re.compile(r"\btrace\b", re.IGNORECASE), "signal:trace"),
            (
                re.compile(r"\ball\s+grounds\b", re.IGNORECASE),
                "signal:all_grounds",
            ),
            (
                re.compile(r"\ball\s+provisions\b", re.IGNORECASE),
                "signal:all_provisions",
            ),
            (
                re.compile(r"\bevery\s+\w+\s+under\b", re.IGNORECASE),
                "signal:every_under",
            ),
            (
                re.compile(r"\bcomprehensive\b", re.IGNORECASE),
                "signal:comprehensive",
            ),
            (re.compile(r"\bexhaustive\b", re.IGNORECASE), "signal:exhaustive"),
            (
                re.compile(r"\binterplay\s+between\b", re.IGNORECASE),
                "signal:interplay_between",
            ),
            (
                re.compile(r"\brelationship\s+between\b", re.IGNORECASE),
                "signal:relationship_between",
            ),
            (
                re.compile(r"how\s+has\s+.+?\s+been\s+interpreted", re.IGNORECASE),
                "signal:interpreted_over_time",
            ),
            (
                re.compile(r"trace\s+the\s+.+?\s+jurisprudence", re.IGNORECASE),
                "signal:trace_jurisprudence",
            ),
            (
                re.compile(r"\bhistory\s+of\b", re.IGNORECASE),
                "signal:history_of",
            ),
            (
                re.compile(r"\bdevelopment\s+of\b", re.IGNORECASE),
                "signal:development_of",
            ),
        ]

        # --- COMPLEX heuristic patterns ---
        self._multi_hop_pattern = re.compile(
            r"\b(?:in\s+light\s+of|read\s+with|read\s+together)\b",
            re.IGNORECASE,
        )
        self._cross_jurisdiction_patterns: list[re.Pattern[str]] = [
            re.compile(
                r"\b(?:Delhi|Mumbai|Kolkata|Chennai|Bangalore|Hyderabad)"
                r"\s+and\s+"
                r"(?:Delhi|Mumbai|Kolkata|Chennai|Bangalore|Hyderabad)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\bstate\s+and\s+central\b", re.IGNORECASE),
            re.compile(r"\bcentral\s+and\s+state\b", re.IGNORECASE),
            re.compile(
                r"\b(?:Supreme\s+Court|High\s+Court|District\s+Court|Tribunal)"
                r"\s+and\s+"
                r"(?:Supreme\s+Court|High\s+Court|District\s+Court|Tribunal)\b",
                re.IGNORECASE,
            ),
        ]

    def classify(self, query_text: str) -> RouterResult:
        """Classify a query into a retrieval route.

        Args:
            query_text: Raw user query.

        Returns:
            RouterResult with route, confidence, and triggering signals.
        """
        try:
            query_lower = query_text.lower().strip()

            # 1. SIMPLE — highest priority for direct lookups
            result = self._check_simple(query_lower)
            if result is not None:
                _log.debug(
                    "router_classified",
                    route=result.route.value,
                    confidence=result.confidence,
                    signals=result.signals,
                )
                return result

            # 2. ANALYTICAL — comparative / historical / exhaustive
            result = self._check_analytical(query_lower)
            if result is not None:
                _log.debug(
                    "router_classified",
                    route=result.route.value,
                    confidence=result.confidence,
                    signals=result.signals,
                )
                return result

            # 3. COMPLEX — multi-act, multi-section, cross-jurisdictional
            result = self._check_complex(query_text)
            if result is not None:
                _log.debug(
                    "router_classified",
                    route=result.route.value,
                    confidence=result.confidence,
                    signals=result.signals,
                )
                return result

            # 4. Default: STANDARD
            _log.debug("router_default_standard", query=query_text[:100])
            return RouterResult(
                route=QueryRoute.STANDARD,
                confidence=0.5,
                signals=["default:standard"],
            )

        except Exception:
            _log.warning(
                "router_classification_error",
                query=query_text[:100],
                exc_info=True,
            )
            return RouterResult(
                route=QueryRoute.STANDARD,
                confidence=0.5,
                signals=["error:fallback_standard"],
            )

    def _check_simple(self, query_lower: str) -> RouterResult | None:
        """Check for direct lookup patterns (SIMPLE route)."""
        for pattern, signal in self._simple_patterns:
            if pattern.search(query_lower):
                return RouterResult(
                    route=QueryRoute.SIMPLE,
                    confidence=1.0,
                    signals=[signal],
                )
        return None

    def _check_analytical(self, query_lower: str) -> RouterResult | None:
        """Check for analytical signals (ANALYTICAL route)."""
        signals: list[str] = []
        for pattern, signal in self._analytical_patterns:
            if pattern.search(query_lower):
                signals.append(signal)

        if signals:
            return RouterResult(
                route=QueryRoute.ANALYTICAL,
                confidence=0.9,
                signals=signals,
            )
        return None

    def _check_complex(self, query_text: str) -> RouterResult | None:
        """Check complexity heuristics (COMPLEX route)."""
        signals: list[str] = []

        # Multi-act: more than 1 act referenced
        acts = extract_act_references(query_text)
        if len(acts) > 1:
            signals.append(f"heuristic:multi_act({len(acts)})")

        # Multi-section: more than 2 section references
        sections = extract_section_references(query_text)
        if len(sections) > 2:
            signals.append(f"heuristic:multi_section({len(sections)})")

        # Cross-jurisdictional references
        for pattern in self._cross_jurisdiction_patterns:
            if pattern.search(query_text):
                signals.append("heuristic:cross_jurisdictional")
                break

        # Multi-hop keywords
        if self._multi_hop_pattern.search(query_text):
            signals.append("heuristic:multi_hop")

        if signals:
            return RouterResult(
                route=QueryRoute.COMPLEX,
                confidence=0.8,
                signals=signals,
            )
        return None
