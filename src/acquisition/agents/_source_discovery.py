from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.acquisition._models import SourceDefinition, SourceRegistry

_log = get_logger(__name__)


class SourceDiscoveryAgent:
    """Loads and validates sources from the config registry.

    Filters to enabled sources and provides them to the pipeline.
    """

    def __init__(self, registry: SourceRegistry) -> None:
        self._registry = registry

    def get_enabled_sources(self) -> list[SourceDefinition]:
        """Return only sources that are marked as enabled."""
        enabled = [s for s in self._registry.sources if s.enabled]
        _log.info(
            "sources_discovered",
            total=len(self._registry.sources),
            enabled=len(enabled),
        )
        return enabled

    def get_source_by_name(self, name: str) -> SourceDefinition | None:
        """Find a source by its display name (case-insensitive)."""
        name_lower = name.lower()
        for source in self._registry.sources:
            if source.name.lower() == name_lower:
                return source
        return None

    def get_source_by_type(self, source_type: str) -> SourceDefinition | None:
        """Find a source by its source_type value."""
        for source in self._registry.sources:
            if source.source_type.value == source_type:
                return source
        return None
