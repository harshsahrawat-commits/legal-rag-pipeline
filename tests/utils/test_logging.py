from __future__ import annotations

from src.utils._logging import configure_logging, get_logger


class TestLogging:
    def test_get_logger_returns_structlog_proxy(self):
        log = get_logger("test.module")
        # Before configure_logging, returns a lazy proxy; after, a BoundLogger.
        # Both are valid structlog loggers with .info(), .error(), etc.
        assert hasattr(log, "info")
        assert hasattr(log, "error")
        assert hasattr(log, "warning")

    def test_get_logger_after_configure_returns_bound(self):
        configure_logging(log_level="INFO", json_output=True)
        log = get_logger("test.configured")
        assert hasattr(log, "info")

    def test_configure_logging_json(self):
        configure_logging(log_level="DEBUG", json_output=True)
        log = get_logger("test")
        assert log is not None

    def test_configure_logging_console(self):
        configure_logging(log_level="INFO", json_output=False)
        log = get_logger("test")
        assert log is not None
