"""Tests for hallucination config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hallucination._config import load_hallucination_config
from src.hallucination._exceptions import HallucinationConfigError
from src.hallucination._models import HallucinationConfig


class TestLoadHallucinationConfig:
    def test_returns_defaults_when_file_missing(self, tmp_path: Path) -> None:
        config = load_hallucination_config(tmp_path / "nonexistent.yaml")
        assert isinstance(config, HallucinationConfig)
        assert config.settings.genground_enabled is True

    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "hall.yaml"
        yaml_path.write_text(
            "settings:\n  genground_enabled: false\n  confidence_warning_threshold: 0.7\n",
            encoding="utf-8",
        )
        config = load_hallucination_config(yaml_path)
        assert config.settings.genground_enabled is False
        assert config.settings.confidence_warning_threshold == 0.7

    def test_loads_empty_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("", encoding="utf-8")
        config = load_hallucination_config(yaml_path)
        assert isinstance(config, HallucinationConfig)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("settings:\n  genground_enabled: {{bad}", encoding="utf-8")
        with pytest.raises(HallucinationConfigError):
            load_hallucination_config(yaml_path)

    def test_defaults_when_no_path(self) -> None:
        # Default path won't exist in test env — should return defaults
        config = load_hallucination_config(Path("configs/nonexistent_test.yaml"))
        assert isinstance(config, HallucinationConfig)

    def test_custom_weights(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "weights.yaml"
        yaml_path.write_text(
            "settings:\n"
            "  weight_retrieval_relevance: 0.30\n"
            "  weight_citation_verification: 0.30\n"
            "  weight_source_authority: 0.10\n"
            "  weight_chunk_agreement: 0.10\n"
            "  weight_source_recency: 0.10\n"
            "  weight_query_specificity: 0.10\n",
            encoding="utf-8",
        )
        config = load_hallucination_config(yaml_path)
        assert config.settings.weight_retrieval_relevance == 0.30

    def test_neo4j_config(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "neo4j.yaml"
        yaml_path.write_text(
            "settings:\n  neo4j_uri: bolt://custom:7688\n  neo4j_user: admin\n",
            encoding="utf-8",
        )
        config = load_hallucination_config(yaml_path)
        assert config.settings.neo4j_uri == "bolt://custom:7688"
        assert config.settings.neo4j_user == "admin"

    def test_ipc_repeal_date(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "temporal.yaml"
        yaml_path.write_text(
            "settings:\n  ipc_repeal_date: '2024-07-01'\n",
            encoding="utf-8",
        )
        config = load_hallucination_config(yaml_path)
        assert config.settings.ipc_repeal_date == "2024-07-01"
