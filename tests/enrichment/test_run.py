from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.enrichment._models import EnrichmentResult
from src.enrichment.run import main


class TestCLI:
    def test_dry_run_exits_0(self):
        mock_result = EnrichmentResult(documents_found=0)
        mock_result.finished_at = mock_result.started_at

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("src.enrichment.run.EnrichmentPipeline", return_value=mock_pipeline),
            patch("sys.argv", ["run.py", "--dry-run", "--console-log"]),
            patch("sys.exit") as mock_exit,
        ):
            main()
            mock_exit.assert_called_once_with(0)

    def test_exits_1_on_errors(self):
        mock_result = EnrichmentResult(
            documents_found=1,
            documents_failed=1,
            errors=["Failed: doc1"],
        )
        mock_result.finished_at = mock_result.started_at

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("src.enrichment.run.EnrichmentPipeline", return_value=mock_pipeline),
            patch("sys.argv", ["run.py", "--console-log"]),
            patch("sys.exit") as mock_exit,
        ):
            main()
            mock_exit.assert_called_once_with(1)

    def test_contextual_stage(self):
        mock_result = EnrichmentResult(documents_found=0)
        mock_result.finished_at = mock_result.started_at

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("src.enrichment.run.EnrichmentPipeline", return_value=mock_pipeline),
            patch(
                "sys.argv",
                ["run.py", "--stage=contextual_retrieval", "--dry-run", "--console-log"],
            ),
            patch("sys.exit"),
        ):
            main()
            call_kwargs = mock_pipeline.run.call_args.kwargs
            assert call_kwargs["stage"] == "contextual_retrieval"
            assert call_kwargs["dry_run"] is True

    def test_quim_stage(self):
        mock_result = EnrichmentResult(documents_found=0)
        mock_result.finished_at = mock_result.started_at

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("src.enrichment.run.EnrichmentPipeline", return_value=mock_pipeline),
            patch("sys.argv", ["run.py", "--stage=quim_rag", "--console-log"]),
            patch("sys.exit"),
        ):
            main()
            call_kwargs = mock_pipeline.run.call_args.kwargs
            assert call_kwargs["stage"] == "quim_rag"

    def test_source_filter_passed(self):
        mock_result = EnrichmentResult(documents_found=0)
        mock_result.finished_at = mock_result.started_at

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("src.enrichment.run.EnrichmentPipeline", return_value=mock_pipeline),
            patch(
                "sys.argv",
                ["run.py", "--source=Indian Kanoon", "--dry-run", "--console-log"],
            ),
            patch("sys.exit"),
        ):
            main()
            call_kwargs = mock_pipeline.run.call_args.kwargs
            assert call_kwargs["source_name"] == "Indian Kanoon"

    def test_custom_config_path(self):
        mock_result = EnrichmentResult(documents_found=0)
        mock_result.finished_at = mock_result.started_at

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with (
            patch("src.enrichment.run.EnrichmentPipeline", return_value=mock_pipeline) as mock_cls,
            patch(
                "sys.argv",
                ["run.py", "--config=/tmp/custom.yaml", "--dry-run", "--console-log"],
            ),
            patch("sys.exit"),
        ):
            main()
            from pathlib import Path

            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["config_path"] == Path("/tmp/custom.yaml")


class TestExports:
    def test_enrichment_pipeline_importable(self):
        from src.enrichment import EnrichmentPipeline

        assert EnrichmentPipeline is not None

    def test_enrichment_config_importable(self):
        from src.enrichment import EnrichmentConfig

        assert EnrichmentConfig is not None

    def test_enrichment_result_importable(self):
        from src.enrichment import EnrichmentResult

        assert EnrichmentResult is not None
