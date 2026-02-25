"""Tests for the embedding CLI and __init__ exports."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestExports:
    def test_exports(self) -> None:
        from src.embedding import EmbeddingConfig, EmbeddingPipeline, EmbeddingResult

        assert EmbeddingConfig is not None
        assert EmbeddingPipeline is not None
        assert EmbeddingResult is not None

    def test_all_list(self) -> None:
        import src.embedding

        assert "EmbeddingPipeline" in src.embedding.__all__
        assert "EmbeddingConfig" in src.embedding.__all__
        assert "EmbeddingResult" in src.embedding.__all__


class TestCLI:
    def test_main_dry_run(self, tmp_path) -> None:
        from src.embedding.run import main

        config = tmp_path / "cfg.yaml"
        config.write_text("settings:\n  input_dir: nonexistent\n", encoding="utf-8")

        with (
            patch("sys.argv", ["run", "--dry-run", "--config", str(config)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0

    def test_main_unknown_source(self, tmp_path) -> None:
        from src.embedding.run import main

        config = tmp_path / "cfg.yaml"
        config.write_text("settings:\n  input_dir: nonexistent\n", encoding="utf-8")

        with (
            patch("sys.argv", ["run", "--source", "BadSource", "--config", str(config)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_main_accepts_device_flag(self, tmp_path) -> None:
        from src.embedding.run import main

        config = tmp_path / "cfg.yaml"
        config.write_text("settings:\n  input_dir: nonexistent\n", encoding="utf-8")

        with (
            patch("sys.argv", ["run", "--dry-run", "--device", "cuda", "--config", str(config)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0
