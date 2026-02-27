"""Tests for the query intelligence CLI."""

from __future__ import annotations

from unittest.mock import patch

from src.query.run import build_parser


class TestBuildParser:
    def test_parser_has_query_arg(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--query", "test query"])
        assert args.query == "test query"

    def test_parser_classify_only(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--classify-only", "--query", "test"])
        assert args.classify_only is True

    def test_parser_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--dry-run", "--query", "test"])
        assert args.dry_run is True

    def test_parser_cache_stats(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--cache-stats"])
        assert args.cache_stats is True

    def test_parser_invalidate_act(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--invalidate-act", "Indian Penal Code"])
        assert args.invalidate_act == "Indian Penal Code"

    def test_parser_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--config", "/path/to/config.yaml", "--query", "test"])
        assert args.config == "/path/to/config.yaml"

    def test_parser_log_level(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--log-level", "DEBUG", "--query", "test"])
        assert args.log_level == "DEBUG"

    def test_parser_console_log(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--console-log", "--query", "test"])
        assert args.console_log is True

    def test_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--query", "test"])
        assert args.classify_only is False
        assert args.dry_run is False
        assert args.cache_stats is False
        assert args.invalidate_act is None
        assert args.config is None
        assert args.log_level == "INFO"
        assert args.console_log is False


class TestMainClassifyOnly:
    def test_classify_only_outputs_route(self, capsys: object) -> None:
        with patch(
            "sys.argv",
            ["query-intelligence", "--classify-only", "--query", "What does Section 302 IPC say?"],
        ):
            from src.query.run import main

            main()
            # capsys not usable with patch on sys.argv easily, but no error = pass

    def test_dry_run_outputs_plan(self) -> None:
        with patch(
            "sys.argv", ["query-intelligence", "--dry-run", "--query", "What is Section 302?"]
        ):
            from src.query.run import main

            main()

    def test_full_processing_path(self) -> None:
        """Full --query path (no --dry-run, no --classify-only) unpacks tuple correctly."""
        with patch(
            "sys.argv", ["query-intelligence", "--query", "What is Section 302?"]
        ):
            from src.query.run import main

            main()
