from __future__ import annotations

from pathlib import Path

import pytest

from src.acquisition._config import load_source_registry
from src.acquisition._models import SourceType
from src.utils._exceptions import ConfigurationError


class TestLoadSourceRegistry:
    def test_load_default_config(self):
        registry = load_source_registry(Path("configs/sources.yaml"))
        assert len(registry.sources) == 2
        assert registry.settings.output_dir == Path("data/raw")

    def test_source_types(self):
        registry = load_source_registry(Path("configs/sources.yaml"))
        types = {s.source_type for s in registry.sources}
        assert types == {SourceType.INDIAN_KANOON, SourceType.INDIA_CODE}

    def test_indian_kanoon_config(self):
        registry = load_source_registry(Path("configs/sources.yaml"))
        ik = next(s for s in registry.sources if s.source_type == SourceType.INDIAN_KANOON)
        assert ik.name == "Indian Kanoon"
        assert ik.rate_limit_requests_per_second == 0.5
        assert len(ik.scrape_config.seed_queries) > 0

    def test_india_code_config(self):
        registry = load_source_registry(Path("configs/sources.yaml"))
        ic = next(s for s in registry.sources if s.source_type == SourceType.INDIA_CODE)
        assert ic.name == "India Code"
        assert ic.rate_limit_requests_per_second == 0.33
        assert ic.scrape_config.max_pages_per_query == 9
        assert ic.scrape_config.max_documents == 50

    def test_missing_file_raises(self):
        with pytest.raises(ConfigurationError, match="not found"):
            load_source_registry(Path("configs/nonexistent.yaml"))

    def test_invalid_yaml_raises(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{ invalid yaml }}", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_source_registry(bad_file)

    def test_invalid_schema_raises(self, tmp_path):
        bad_file = tmp_path / "bad_schema.yaml"
        bad_file.write_text("sources: not_a_list", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="validation failed"):
            load_source_registry(bad_file)

    def test_non_dict_yaml_raises(self, tmp_path):
        bad_file = tmp_path / "list.yaml"
        bad_file.write_text("- item1\n- item2", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            load_source_registry(bad_file)
