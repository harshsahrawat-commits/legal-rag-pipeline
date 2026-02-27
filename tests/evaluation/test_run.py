"""Tests for the evaluation CLI (run.py)."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from src.evaluation.run import _build_parser, _run

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def parser() -> argparse.ArgumentParser:
    return _build_parser()


class TestBuildParser:
    def test_parser_type(self, parser: argparse.ArgumentParser) -> None:
        assert isinstance(parser, argparse.ArgumentParser)

    def test_queries_arg(self, parser: argparse.ArgumentParser) -> None:
        from pathlib import Path

        args = parser.parse_args(["--queries", "data/eval/test_queries.json"])
        assert args.queries == Path("data/eval/test_queries.json")

    def test_query_arg(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--query", "What is Section 302?"])
        assert args.query == "What is Section 302?"

    def test_dry_run_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--queries", "test.json", "--dry-run"])
        assert args.dry_run is True

    def test_skip_ragas_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--queries", "test.json", "--skip-ragas"])
        assert args.skip_ragas is True

    def test_human_generate_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--queries", "test.json", "--human-generate"])
        assert args.human_generate is True

    def test_human_import_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--human-import"])
        assert args.human_import is True

    def test_report_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--queries", "test.json", "--report"])
        assert args.report is True

    def test_config_arg(self, parser: argparse.ArgumentParser) -> None:
        from pathlib import Path

        args = parser.parse_args(["--config", "custom.yaml", "--queries", "test.json"])
        assert args.config == Path("custom.yaml")

    def test_console_log_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["--console-log", "--queries", "test.json"])
        assert args.console_log is True

    def test_defaults(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args([])
        assert args.queries is None
        assert args.query is None
        assert args.dry_run is False
        assert args.skip_ragas is False
        assert args.log_level == "INFO"


class TestRunDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_no_queries(self) -> None:
        args = argparse.Namespace(
            dry_run=True,
            queries=None,
            query=None,
            config=None,
            skip_ragas=False,
            human_generate=False,
            human_import=False,
            report=False,
        )
        result = await _run(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_dry_run_with_queries(self, test_queries_path: Path) -> None:
        args = argparse.Namespace(
            dry_run=True,
            queries=test_queries_path,
            query=None,
            config=None,
            skip_ragas=False,
            human_generate=False,
            human_import=False,
            report=False,
        )
        result = await _run(args)
        assert result == 0


class TestRunBatchMode:
    @pytest.mark.asyncio
    async def test_batch_evaluation(self, test_queries_path: Path) -> None:
        args = argparse.Namespace(
            dry_run=False,
            queries=test_queries_path,
            query=None,
            config=None,
            skip_ragas=True,
            human_generate=False,
            human_import=False,
            report=False,
        )
        result = await _run(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_batch_with_report(self, test_queries_path: Path, tmp_path: Path) -> None:
        args = argparse.Namespace(
            dry_run=False,
            queries=test_queries_path,
            query=None,
            config=None,
            skip_ragas=True,
            human_generate=False,
            human_import=False,
            report=True,
        )
        report_path = tmp_path / "report.json"
        with patch(
            "src.evaluation._report.EvaluationReporter.save",
            return_value=report_path,
        ):
            result = await _run(args)
            assert result == 0


class TestRunSingleQueryMode:
    @pytest.mark.asyncio
    async def test_single_query_evaluation(self) -> None:
        args = argparse.Namespace(
            dry_run=False,
            queries=None,
            query="What is Section 302 IPC?",
            response="Section 302 prescribes punishment for murder.",
            contexts=None,
            config=None,
            skip_ragas=True,
            human_generate=False,
            human_import=False,
            report=False,
        )
        result = await _run(args)
        assert result == 0

    @pytest.mark.asyncio
    async def test_single_query_no_response_returns_1(self) -> None:
        args = argparse.Namespace(
            dry_run=False,
            queries=None,
            query="What is Section 302?",
            response=None,
            contexts=None,
            config=None,
            skip_ragas=True,
            human_generate=False,
            human_import=False,
            report=False,
        )
        result = await _run(args)
        assert result == 1

    @pytest.mark.asyncio
    async def test_single_query_with_contexts(self, tmp_path: Path) -> None:
        contexts_file = tmp_path / "contexts.json"
        contexts_file.write_text(
            json.dumps(["Section 302. Punishment for murder."]),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            dry_run=False,
            queries=None,
            query="What is Section 302 IPC?",
            response="Section 302 prescribes punishment for murder.",
            contexts=contexts_file,
            config=None,
            skip_ragas=True,
            human_generate=False,
            human_import=False,
            report=False,
        )
        result = await _run(args)
        assert result == 0


class TestRunNoInput:
    @pytest.mark.asyncio
    async def test_no_input_returns_1(self) -> None:
        args = argparse.Namespace(
            dry_run=False,
            queries=None,
            query=None,
            config=None,
            skip_ragas=False,
            human_generate=False,
            human_import=False,
            report=False,
        )
        result = await _run(args)
        assert result == 1


class TestRunHumanGenerate:
    @pytest.mark.asyncio
    async def test_human_generate_requires_queries(self) -> None:
        args = argparse.Namespace(
            dry_run=False,
            queries=None,
            query=None,
            config=None,
            skip_ragas=False,
            human_generate=True,
            human_import=False,
            report=False,
        )
        result = await _run(args)
        assert result == 1

    @pytest.mark.asyncio
    async def test_human_generate_with_queries(
        self, test_queries_path: Path, tmp_path: Path
    ) -> None:
        args = argparse.Namespace(
            dry_run=False,
            queries=test_queries_path,
            query=None,
            config=None,
            skip_ragas=False,
            human_generate=True,
            human_import=False,
            report=False,
        )
        # Patch settings to use tmp_path for worksheets
        with patch("src.evaluation._config.load_evaluation_config") as mock_config:
            from src.evaluation._models import EvaluationConfig, EvaluationSettings

            mock_config.return_value = EvaluationConfig(
                settings=EvaluationSettings(worksheets_dir=str(tmp_path / "worksheets"))
            )
            result = await _run(args)
            assert result == 0


class TestRunHumanImport:
    @pytest.mark.asyncio
    async def test_human_import_with_scoresheets(self, tmp_path: Path) -> None:
        # Create scoresheet
        scoresheet = tmp_path / "scores.json"
        scoresheet.write_text(
            json.dumps(
                [
                    {
                        "query_id": "q1",
                        "evaluator_id": "eval1",
                        "accuracy": 4,
                        "completeness": 4,
                        "recency": 3,
                        "usefulness": 5,
                    }
                ]
            ),
            encoding="utf-8",
        )

        args = argparse.Namespace(
            dry_run=False,
            queries=None,
            query=None,
            config=None,
            skip_ragas=False,
            human_generate=False,
            human_import=True,
            report=False,
        )
        with patch("src.evaluation._config.load_evaluation_config") as mock_config:
            from src.evaluation._models import EvaluationConfig, EvaluationSettings

            mock_config.return_value = EvaluationConfig(
                settings=EvaluationSettings(scoresheets_dir=str(tmp_path))
            )
            result = await _run(args)
            assert result == 0
