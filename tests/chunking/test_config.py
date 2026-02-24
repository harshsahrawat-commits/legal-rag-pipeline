from __future__ import annotations

from pathlib import Path

import pytest

from src.chunking._config import load_chunking_config
from src.chunking._models import ChunkingConfig
from src.utils._exceptions import ConfigurationError


class TestLoadChunkingConfig:
    def test_load_default_config(self):
        """configs/chunking.yaml should load and validate."""
        config = load_chunking_config(Path("configs/chunking.yaml"))
        assert isinstance(config, ChunkingConfig)
        assert config.settings.max_tokens == 1500

    def test_load_custom_config(self, tmp_path: Path):
        cfg_file = tmp_path / "custom.yaml"
        cfg_file.write_text(
            "settings:\n  max_tokens: 1000\n  overlap_tokens: 100\n",
            encoding="utf-8",
        )
        config = load_chunking_config(cfg_file)
        assert config.settings.max_tokens == 1000
        assert config.settings.overlap_tokens == 100

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_chunking_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("{{invalid", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_chunking_config(cfg_file)

    def test_non_mapping_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            load_chunking_config(cfg_file)

    def test_invalid_values_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "bad_values.yaml"
        cfg_file.write_text(
            "settings:\n  max_tokens: not_a_number\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="Config validation failed"):
            load_chunking_config(cfg_file)

    def test_empty_mapping_uses_defaults(self, tmp_path: Path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("settings: {}\n", encoding="utf-8")
        config = load_chunking_config(cfg_file)
        assert config.settings.max_tokens == 1500
