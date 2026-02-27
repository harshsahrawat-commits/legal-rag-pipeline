"""Tests for evaluation config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation._config import load_evaluation_config
from src.evaluation._exceptions import EvaluationConfigError
from src.evaluation._models import EvaluationConfig, EvaluationSettings


class TestLoadEvaluationConfig:
    """Test load_evaluation_config function."""

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        config = load_evaluation_config(tmp_path / "nonexistent.yaml")
        assert isinstance(config, EvaluationConfig)
        assert config.settings.ragas_enabled is True

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "eval.yaml"
        config_file.write_text(
            "settings:\n  ragas_enabled: false\n  latency_simple_ms: 100.0\n",
            encoding="utf-8",
        )
        config = load_evaluation_config(config_file)
        assert config.settings.ragas_enabled is False
        assert config.settings.latency_simple_ms == 100.0

    def test_load_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "eval.yaml"
        config_file.write_text("", encoding="utf-8")
        config = load_evaluation_config(config_file)
        assert isinstance(config, EvaluationConfig)

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "eval.yaml"
        config_file.write_text(
            "settings:\n  ragas_enabled: [invalid\n",
            encoding="utf-8",
        )
        with pytest.raises(EvaluationConfigError):
            load_evaluation_config(config_file)

    def test_load_default_path_fallback(self) -> None:
        # Default path (configs/evaluation.yaml) should not raise even if missing
        config = load_evaluation_config(Path("configs/nonexistent_eval.yaml"))
        assert isinstance(config, EvaluationConfig)

    def test_override_all_settings(self, tmp_path: Path) -> None:
        config_file = tmp_path / "eval.yaml"
        config_file.write_text(
            "settings:\n"
            "  ragas_enabled: false\n"
            "  ragas_batch_size: 5\n"
            "  latency_simple_ms: 100.0\n"
            "  latency_standard_ms: 400.0\n"
            "  cache_hit_target: 0.50\n"
            "  human_eval_enabled: true\n"
            "  report_format: text\n",
            encoding="utf-8",
        )
        config = load_evaluation_config(config_file)
        s = config.settings
        assert s.ragas_enabled is False
        assert s.ragas_batch_size == 5
        assert s.latency_simple_ms == 100.0
        assert s.latency_standard_ms == 400.0
        assert s.cache_hit_target == 0.50
        assert s.human_eval_enabled is True
        assert s.report_format == "text"

    def test_load_none_path_uses_default(self) -> None:
        # Passing None should use default path, which returns defaults if not found
        config = load_evaluation_config(None)
        assert isinstance(config, EvaluationConfig)

    def test_returns_evaluation_config_type(self, tmp_path: Path) -> None:
        config_file = tmp_path / "eval.yaml"
        config_file.write_text("settings:\n  ragas_enabled: true\n", encoding="utf-8")
        config = load_evaluation_config(config_file)
        assert isinstance(config.settings, EvaluationSettings)
