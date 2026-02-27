"""Tests for query intelligence config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.query._config import load_query_config
from src.query._models import QueryConfig
from src.utils._exceptions import ConfigurationError


class TestLoadQueryConfig:
    """Tests for load_query_config()."""

    def test_load_default_config(self) -> None:
        """Load the actual configs/query.yaml file."""
        config = load_query_config(Path("configs/query.yaml"))
        assert isinstance(config, QueryConfig)
        assert config.settings.cache_enabled is True
        assert config.settings.cache_similarity_threshold == 0.92

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_query_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{invalid yaml::", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_query_config(bad_file)

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "list.yaml"
        bad_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            load_query_config(bad_file)

    def test_valid_custom_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "query.yaml"
        config_file.write_text(
            "settings:\n"
            "  cache_enabled: false\n"
            "  cache_similarity_threshold: 0.85\n"
            "  hyde_enabled: false\n",
            encoding="utf-8",
        )
        config = load_query_config(config_file)
        assert config.settings.cache_enabled is False
        assert config.settings.cache_similarity_threshold == 0.85
        assert config.settings.hyde_enabled is False

    def test_empty_settings_uses_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "query.yaml"
        config_file.write_text("settings: {}\n", encoding="utf-8")
        config = load_query_config(config_file)
        assert config.settings.cache_enabled is True
        assert config.settings.cache_ttl_seconds == 86400

    def test_partial_settings_merges_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "query.yaml"
        config_file.write_text(
            "settings:\n  device: cuda\n",
            encoding="utf-8",
        )
        config = load_query_config(config_file)
        assert config.settings.device == "cuda"
        assert config.settings.cache_enabled is True  # default preserved

    def test_validation_error_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "query.yaml"
        # cache_similarity_threshold should be a float, not a string
        config_file.write_text(
            "settings:\n  cache_similarity_threshold: not_a_number\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="Config validation failed"):
            load_query_config(config_file)
