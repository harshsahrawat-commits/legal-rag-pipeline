from __future__ import annotations

from pathlib import Path

import pytest

from src.embedding._config import load_embedding_config
from src.embedding._models import EmbeddingConfig
from src.utils._exceptions import ConfigurationError


class TestLoadEmbeddingConfig:
    def test_load_default_config(self) -> None:
        """Load configs/embedding.yaml (committed to repo)."""
        config = load_embedding_config(Path("configs/embedding.yaml"))
        assert isinstance(config, EmbeddingConfig)
        assert config.settings.model_name_or_path == "BAAI/bge-m3"
        assert config.settings.embedding_dim == 768

    def test_load_custom_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.write_text(
            'settings:\n  model_name_or_path: "test/model"\n  batch_size: 64\n',
            encoding="utf-8",
        )
        config = load_embedding_config(cfg)
        assert config.settings.model_name_or_path == "test/model"
        assert config.settings.batch_size == 64

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_embedding_config(tmp_path / "nope.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("{{invalid", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_embedding_config(cfg)

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            load_embedding_config(cfg)

    def test_invalid_field_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad_field.yaml"
        cfg.write_text(
            "settings:\n  embedding_dim: not_a_number\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="Config validation failed"):
            load_embedding_config(cfg)

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("{}\n", encoding="utf-8")
        config = load_embedding_config(cfg)
        assert config.settings.embedding_dim == 768

    def test_partial_settings(self, tmp_path: Path) -> None:
        cfg = tmp_path / "partial.yaml"
        cfg.write_text("settings:\n  device: cuda\n", encoding="utf-8")
        config = load_embedding_config(cfg)
        assert config.settings.device == "cuda"
        assert config.settings.embedding_dim == 768  # default preserved
