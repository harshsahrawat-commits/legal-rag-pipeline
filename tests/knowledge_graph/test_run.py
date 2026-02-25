"""Tests for the CLI entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.knowledge_graph._models import KGResult


class TestMain:
    def test_dry_run_exits_zero(self, tmp_path: Path) -> None:
        """--dry-run with empty dir should exit 0."""
        from src.knowledge_graph.run import main

        mock_result = KGResult(documents_found=0)
        mock_result.finished_at = mock_result.started_at

        mock_pipeline_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        mock_pipeline_class.return_value = mock_instance

        with (
            patch("src.knowledge_graph.run.KnowledgeGraphPipeline", mock_pipeline_class),
            patch("src.knowledge_graph.run.configure_logging"),
            patch("sys.argv", ["run", "--dry-run"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0

    def test_errors_exit_one(self) -> None:
        from src.knowledge_graph.run import main

        mock_result = KGResult(errors=["something failed"])

        mock_pipeline_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        mock_pipeline_class.return_value = mock_instance

        with (
            patch("src.knowledge_graph.run.KnowledgeGraphPipeline", mock_pipeline_class),
            patch("src.knowledge_graph.run.configure_logging"),
            patch("sys.argv", ["run", "--dry-run"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    def test_source_arg_passed(self) -> None:
        from src.knowledge_graph.run import main

        mock_result = KGResult()
        mock_pipeline_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        mock_pipeline_class.return_value = mock_instance

        with (
            patch("src.knowledge_graph.run.KnowledgeGraphPipeline", mock_pipeline_class),
            patch("src.knowledge_graph.run.configure_logging"),
            patch("sys.argv", ["run", "--source", "Indian Kanoon", "--dry-run"]),
            pytest.raises(SystemExit),
        ):
            main()

        mock_instance.run.assert_called_once_with(
            source_name="Indian Kanoon",
            dry_run=True,
            skip_integrity=False,
        )

    def test_skip_integrity_arg(self) -> None:
        from src.knowledge_graph.run import main

        mock_result = KGResult()
        mock_pipeline_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        mock_pipeline_class.return_value = mock_instance

        with (
            patch("src.knowledge_graph.run.KnowledgeGraphPipeline", mock_pipeline_class),
            patch("src.knowledge_graph.run.configure_logging"),
            patch("sys.argv", ["run", "--skip-integrity", "--dry-run"]),
            pytest.raises(SystemExit),
        ):
            main()

        mock_instance.run.assert_called_once_with(
            source_name=None,
            dry_run=True,
            skip_integrity=True,
        )

    def test_config_path_arg(self) -> None:
        from src.knowledge_graph.run import main

        mock_result = KGResult()
        mock_pipeline_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        mock_pipeline_class.return_value = mock_instance

        with (
            patch("src.knowledge_graph.run.KnowledgeGraphPipeline", mock_pipeline_class),
            patch("src.knowledge_graph.run.configure_logging"),
            patch("sys.argv", ["run", "--config", "custom.yaml", "--dry-run"]),
            pytest.raises(SystemExit),
        ):
            main()

        mock_pipeline_class.assert_called_once_with(config_path=Path("custom.yaml"))

    def test_console_log_flag(self) -> None:
        from src.knowledge_graph.run import main

        mock_result = KGResult()
        mock_pipeline_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        mock_pipeline_class.return_value = mock_instance

        mock_configure = MagicMock()

        with (
            patch("src.knowledge_graph.run.KnowledgeGraphPipeline", mock_pipeline_class),
            patch("src.knowledge_graph.run.configure_logging", mock_configure),
            patch("sys.argv", ["run", "--console-log", "--dry-run"]),
            pytest.raises(SystemExit),
        ):
            main()

        mock_configure.assert_called_once_with(log_level="INFO", json_output=False)
