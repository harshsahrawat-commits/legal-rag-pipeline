"""Tests for CLI entrypoint (run.py)."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.hallucination.run import _build_parser, _run


class TestBuildParser:
    def test_response_arg(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--response", "test text"])
        assert args.response == "test text"
        assert args.response_file is None

    def test_response_file_arg(self, tmp_path: Path) -> None:
        f = tmp_path / "resp.txt"
        f.write_text("file content")
        parser = _build_parser()
        args = parser.parse_args(["--response-file", str(f)])
        assert args.response_file == f
        assert args.response is None

    def test_mutually_exclusive(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--response", "a", "--response-file", "b"])

    def test_dry_run_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--response", "x", "--dry-run"])
        assert args.dry_run is True

    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--response", "x"])
        assert args.log_level == "INFO"
        assert args.console_log is False
        assert args.config is None
        assert args.dry_run is False


class TestRunFunction:
    async def test_dry_run(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--response", "Section 420 IPC.", "--dry-run"])
        code = await _run(args)
        assert code == 0

    async def test_empty_response(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--response", "   "])
        code = await _run(args)
        assert code == 1

    async def test_full_run_genground_disabled(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--response", "Section 420 of the Indian Penal Code."])
        code = await _run(args)
        assert code == 0

    async def test_response_file(self, tmp_path: Path) -> None:
        f = tmp_path / "resp.txt"
        f.write_text("Legal response text.", encoding="utf-8")
        parser = _build_parser()
        args = parser.parse_args(["--response-file", str(f)])
        code = await _run(args)
        assert code == 0


class TestCLIEntrypoint:
    def test_module_invocation_dry_run(self) -> None:
        """python -m src.hallucination --response '...' --dry-run exits 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.hallucination",
                "--response",
                "Section 420 of IPC provides for punishment of cheating.",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
