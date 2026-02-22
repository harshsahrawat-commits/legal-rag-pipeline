from __future__ import annotations

from pathlib import Path

import pytest

from src.parsing._config import load_parsing_config
from src.parsing._models import ParsingConfig
from src.utils._exceptions import ConfigurationError


class TestLoadParsingConfig:
    """Config loading tests."""

    def test_load_default_config(self) -> None:
        """Load the real configs/parsing.yaml."""
        config = load_parsing_config()
        assert isinstance(config, ParsingConfig)
        assert config.settings.prefer_docling is True
        assert config.settings.ocr_languages == ["eng", "hin"]

    def test_load_custom_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "parsing.yaml"
        config_file.write_text(
            "settings:\n"
            "  input_dir: custom/input\n"
            "  output_dir: custom/output\n"
            "  prefer_docling: false\n"
            "  min_text_completeness: 0.7\n",
            encoding="utf-8",
        )
        config = load_parsing_config(config_file)
        assert config.settings.input_dir == Path("custom/input")
        assert config.settings.prefer_docling is False
        assert config.settings.min_text_completeness == 0.7

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_parsing_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(":\n  invalid: [yaml", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_parsing_config(config_file)

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            load_parsing_config(config_file)

    def test_invalid_field_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad_field.yaml"
        config_file.write_text(
            "settings:\n  min_text_completeness: not_a_number\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigurationError, match="Config validation failed"):
            load_parsing_config(config_file)

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("{}", encoding="utf-8")
        config = load_parsing_config(config_file)
        assert isinstance(config, ParsingConfig)
        assert config.settings.prefer_docling is True


class TestExceptionHierarchy:
    """Verify parsing exceptions extend LegalRAGError."""

    def test_parsing_error_hierarchy(self) -> None:
        from src.parsing._exceptions import (
            DocumentStructureError,
            ParserNotAvailableError,
            ParsingError,
            PDFDownloadError,
            QualityValidationError,
            UnsupportedFormatError,
        )
        from src.utils._exceptions import LegalRAGError

        # All extend ParsingError
        assert issubclass(PDFDownloadError, ParsingError)
        assert issubclass(DocumentStructureError, ParsingError)
        assert issubclass(QualityValidationError, ParsingError)
        assert issubclass(UnsupportedFormatError, ParsingError)
        assert issubclass(ParserNotAvailableError, ParsingError)

        # ParsingError extends LegalRAGError
        assert issubclass(ParsingError, LegalRAGError)

    def test_exceptions_are_catchable(self) -> None:
        from src.parsing._exceptions import ParsingError, PDFDownloadError
        from src.utils._exceptions import LegalRAGError

        exc = PDFDownloadError("test")
        assert isinstance(exc, PDFDownloadError)
        assert isinstance(exc, ParsingError)
        assert isinstance(exc, LegalRAGError)
        assert str(exc) == "test"
