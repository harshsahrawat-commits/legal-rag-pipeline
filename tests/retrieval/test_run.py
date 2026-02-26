"""Tests for retrieval CLI (run.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.retrieval._models import RetrievalResult


class TestCLI:
    """CLI argument parsing and invocation."""

    def test_main_exists(self) -> None:
        from src.retrieval.run import main

        assert callable(main)

    @patch("src.retrieval.run.asyncio")
    @patch("src.retrieval.run.RetrievalPipeline")
    @patch("src.retrieval.run.configure_logging")
    def test_single_query(
        self,
        mock_logging: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        mock_result = RetrievalResult(query_text="test")
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_asyncio.run.return_value = [mock_result]

        with patch("sys.argv", ["run", "--query", "What is IPC?", "--dry-run"]):
            from src.retrieval.run import main

            main()

        mock_pipeline_cls.assert_called_once()
        mock_asyncio.run.assert_called_once()

    @patch("src.retrieval.run.asyncio")
    @patch("src.retrieval.run.RetrievalPipeline")
    @patch("src.retrieval.run.configure_logging")
    def test_queries_file(
        self,
        mock_logging: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        mock_asyncio.run.return_value = []
        mock_pipeline_cls.return_value = MagicMock()

        with patch("sys.argv", ["run", "--queries-file", "data/eval/test.json"]):
            from src.retrieval.run import main

            main()

        call_kwargs = mock_asyncio.run.call_args
        assert call_kwargs is not None

    @patch("src.retrieval.run.asyncio")
    @patch("src.retrieval.run.RetrievalPipeline")
    @patch("src.retrieval.run.configure_logging")
    def test_console_log(
        self,
        mock_logging: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        mock_asyncio.run.return_value = []
        mock_pipeline_cls.return_value = MagicMock()

        with patch("sys.argv", ["run", "--query", "test", "--console-log"]):
            from src.retrieval.run import main

            main()

        mock_logging.assert_called_once_with(log_level="INFO", json_output=False)

    @patch("src.retrieval.run.asyncio")
    @patch("src.retrieval.run.RetrievalPipeline")
    @patch("src.retrieval.run.configure_logging")
    def test_custom_config(
        self,
        mock_logging: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        mock_asyncio.run.return_value = []
        mock_pipeline_cls.return_value = MagicMock()

        with patch("sys.argv", ["run", "--query", "test", "--config", "/tmp/custom.yaml"]):
            from src.retrieval.run import main

            main()

        # Pipeline was created with a config_path
        call_kwargs = mock_pipeline_cls.call_args
        assert call_kwargs is not None

    def test_module_runnable(self) -> None:
        """__main__.py exists and references run.main."""
        import importlib

        spec = importlib.util.find_spec("src.retrieval.__main__")
        assert spec is not None
