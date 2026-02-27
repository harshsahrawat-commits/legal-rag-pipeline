"""Configuration loading for the evaluation module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation._exceptions import EvaluationConfigError
from src.evaluation._models import EvaluationConfig
from src.utils._logging import get_logger

_log = get_logger(__name__)

_DEFAULT_CONFIG_PATH = Path("configs/evaluation.yaml")


def load_evaluation_config(
    config_path: Path | None = None,
) -> EvaluationConfig:
    """Load evaluation config from a YAML file.

    Args:
        config_path: Path to YAML config. Defaults to configs/evaluation.yaml.

    Returns:
        Parsed EvaluationConfig.

    Raises:
        EvaluationConfigError: If the file cannot be read or parsed.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    if not path.exists():
        _log.info("evaluation_config_not_found", path=str(path))
        return EvaluationConfig()

    try:
        import yaml

        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return EvaluationConfig(**raw)
    except ImportError:
        _log.info("pyyaml_not_installed_using_defaults")
        return EvaluationConfig()
    except Exception as exc:
        msg = f"Failed to load evaluation config from {path}: {exc}"
        raise EvaluationConfigError(msg) from exc
