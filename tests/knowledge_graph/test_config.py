from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge_graph._config import load_kg_config
from src.knowledge_graph._models import KGConfig
from src.utils._exceptions import ConfigurationError


class TestLoadKGConfig:
    def test_load_default_config(self) -> None:
        """Load configs/knowledge_graph.yaml (committed to repo)."""
        config = load_kg_config(Path("configs/knowledge_graph.yaml"))
        assert isinstance(config, KGConfig)
        assert config.settings.neo4j_uri == "bolt://localhost:7687"
        assert config.settings.batch_size == 100

    def test_load_custom_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.write_text(
            'settings:\n  neo4j_uri: "bolt://remote:7687"\n  batch_size: 50\n',
            encoding="utf-8",
        )
        config = load_kg_config(cfg)
        assert config.settings.neo4j_uri == "bolt://remote:7687"
        assert config.settings.batch_size == 50

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_kg_config(tmp_path / "nope.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("{{invalid", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_kg_config(cfg)

    def test_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            load_kg_config(cfg)

    def test_invalid_field_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad_field.yaml"
        cfg.write_text(
            "settings:\n  batch_size: not_a_number\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="Config validation failed"):
            load_kg_config(cfg)

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("{}\n", encoding="utf-8")
        config = load_kg_config(cfg)
        assert config.settings.batch_size == 100

    def test_partial_settings(self, tmp_path: Path) -> None:
        cfg = tmp_path / "partial.yaml"
        cfg.write_text('settings:\n  neo4j_database: "testdb"\n', encoding="utf-8")
        config = load_kg_config(cfg)
        assert config.settings.neo4j_database == "testdb"
        assert config.settings.batch_size == 100  # default preserved
