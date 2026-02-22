from __future__ import annotations

import pytest

from src.utils._exceptions import ConfigurationError, LegalRAGError, ValidationError


class TestExceptionHierarchy:
    def test_legal_rag_error_is_exception(self):
        assert issubclass(LegalRAGError, Exception)

    def test_configuration_error_inherits(self):
        assert issubclass(ConfigurationError, LegalRAGError)

    def test_validation_error_inherits(self):
        assert issubclass(ValidationError, LegalRAGError)

    def test_can_raise_and_catch_as_base(self):
        with pytest.raises(LegalRAGError):
            raise ConfigurationError("bad config")

    def test_message_preserved(self):
        err = ValidationError("field X is required")
        assert str(err) == "field X is required"
