from __future__ import annotations

from src.acquisition._models import (
    SourceDefinition,
    SourceRegistry,
    SourceType,
)
from src.acquisition.agents._source_discovery import SourceDiscoveryAgent


def _make_registry(sources=None):
    if sources is None:
        sources = [
            SourceDefinition(
                name="Indian Kanoon",
                source_type=SourceType.INDIAN_KANOON,
                base_url="https://indiankanoon.org",
                enabled=True,
            ),
            SourceDefinition(
                name="India Code",
                source_type=SourceType.INDIA_CODE,
                base_url="https://www.indiacode.nic.in",
                enabled=True,
            ),
        ]
    return SourceRegistry(sources=sources)


class TestSourceDiscoveryAgent:
    def test_get_enabled_sources_all_enabled(self):
        agent = SourceDiscoveryAgent(_make_registry())
        enabled = agent.get_enabled_sources()
        assert len(enabled) == 2

    def test_get_enabled_sources_filters_disabled(self):
        sources = [
            SourceDefinition(
                name="Indian Kanoon",
                source_type=SourceType.INDIAN_KANOON,
                base_url="https://indiankanoon.org",
                enabled=True,
            ),
            SourceDefinition(
                name="India Code",
                source_type=SourceType.INDIA_CODE,
                base_url="https://www.indiacode.nic.in",
                enabled=False,
            ),
        ]
        agent = SourceDiscoveryAgent(_make_registry(sources))
        enabled = agent.get_enabled_sources()
        assert len(enabled) == 1
        assert enabled[0].name == "Indian Kanoon"

    def test_get_source_by_name(self):
        agent = SourceDiscoveryAgent(_make_registry())
        result = agent.get_source_by_name("Indian Kanoon")
        assert result is not None
        assert result.source_type == SourceType.INDIAN_KANOON

    def test_get_source_by_name_case_insensitive(self):
        agent = SourceDiscoveryAgent(_make_registry())
        result = agent.get_source_by_name("indian kanoon")
        assert result is not None

    def test_get_source_by_name_not_found(self):
        agent = SourceDiscoveryAgent(_make_registry())
        result = agent.get_source_by_name("Nonexistent")
        assert result is None

    def test_get_source_by_type(self):
        agent = SourceDiscoveryAgent(_make_registry())
        result = agent.get_source_by_type("india_code")
        assert result is not None
        assert result.name == "India Code"

    def test_get_source_by_type_not_found(self):
        agent = SourceDiscoveryAgent(_make_registry())
        result = agent.get_source_by_type("nonexistent")
        assert result is None
