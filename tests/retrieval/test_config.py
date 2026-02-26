"""Tests for retrieval config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval._config import load_retrieval_config
from src.retrieval._models import RetrievalConfig, RetrievalSettings
from src.utils._exceptions import ConfigurationError


class TestLoadRetrievalConfig:
    """Config loader tests."""

    def test_load_default_config(self) -> None:
        """Load the real configs/retrieval.yaml."""
        config = load_retrieval_config(Path("configs/retrieval.yaml"))
        assert isinstance(config, RetrievalConfig)
        assert isinstance(config.settings, RetrievalSettings)
        assert config.settings.qdrant_host == "localhost"

    def test_load_custom_config(self, tmp_path: Path) -> None:
        """Load a custom config file."""
        yaml_content = "settings:\n  qdrant_host: custom-host\n  rerank_top_k: 5\n"
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = load_retrieval_config(config_file)
        assert config.settings.qdrant_host == "custom-host"
        assert config.settings.rerank_top_k == 5
        # Defaults still apply for unset fields
        assert config.settings.dense_fast_top_k == 1000

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_retrieval_config(Path("nonexistent.yaml"))

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{invalid yaml: [")

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_retrieval_config(bad_yaml)

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n")

        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            load_retrieval_config(list_yaml)

    def test_invalid_field_type_raises(self, tmp_path: Path) -> None:
        bad_field = tmp_path / "bad_field.yaml"
        bad_field.write_text("settings:\n  qdrant_port: not_a_number\n")

        with pytest.raises(ConfigurationError, match="Config validation failed"):
            load_retrieval_config(bad_field)

    def test_empty_settings_uses_defaults(self, tmp_path: Path) -> None:
        """An empty settings block should produce all defaults."""
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text("settings: {}\n")

        config = load_retrieval_config(minimal)
        assert config.settings.qdrant_host == "localhost"
        assert config.settings.flare_enabled is True

    def test_empty_file_uses_defaults(self, tmp_path: Path) -> None:
        """A YAML file with just `{}` should produce all defaults."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("{}\n")

        config = load_retrieval_config(empty)
        assert isinstance(config.settings, RetrievalSettings)
