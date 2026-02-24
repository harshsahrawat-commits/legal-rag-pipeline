from __future__ import annotations

from pathlib import Path

import pytest

from src.enrichment._config import load_enrichment_config
from src.enrichment._models import EnrichmentConfig
from src.utils._exceptions import ConfigurationError


class TestLoadEnrichmentConfig:
    def test_load_default_config(self):
        """configs/enrichment.yaml should load and validate."""
        config = load_enrichment_config(Path("configs/enrichment.yaml"))
        assert isinstance(config, EnrichmentConfig)
        assert config.settings.model == "claude-haiku-4-5-20251001"
        assert config.settings.concurrency == 5

    def test_load_custom_config(self, tmp_path: Path):
        cfg_file = tmp_path / "custom.yaml"
        cfg_file.write_text(
            "settings:\n  model: custom-model\n  concurrency: 10\n",
            encoding="utf-8",
        )
        config = load_enrichment_config(cfg_file)
        assert config.settings.model == "custom-model"
        assert config.settings.concurrency == 10

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_enrichment_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("{{invalid", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_enrichment_config(cfg_file)

    def test_non_mapping_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            load_enrichment_config(cfg_file)

    def test_invalid_values_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "bad_values.yaml"
        cfg_file.write_text(
            "settings:\n  concurrency: not_a_number\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="Config validation failed"):
            load_enrichment_config(cfg_file)

    def test_empty_mapping_uses_defaults(self, tmp_path: Path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("settings: {}\n", encoding="utf-8")
        config = load_enrichment_config(cfg_file)
        assert config.settings.model == "claude-haiku-4-5-20251001"
        assert config.settings.concurrency == 5

    def test_partial_settings_preserves_defaults(self, tmp_path: Path):
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text(
            "settings:\n  quim_questions_per_chunk: 3\n",
            encoding="utf-8",
        )
        config = load_enrichment_config(cfg_file)
        assert config.settings.quim_questions_per_chunk == 3
        assert config.settings.model == "claude-haiku-4-5-20251001"  # Default preserved
