"""Configuration loader for the Query Intelligence Layer."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.query._models import QueryConfig
from src.utils._exceptions import ConfigurationError

_DEFAULT_CONFIG_PATH = Path("configs/query.yaml")


def load_query_config(config_path: Path | None = None) -> QueryConfig:
    """Load and validate the query intelligence config from a YAML file.

    Args:
        config_path: Path to the YAML config. Defaults to configs/query.yaml.

    Returns:
        Validated QueryConfig model.

    Raises:
        ConfigurationError: If the file is missing, unreadable, or invalid.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    if not path.exists():
        msg = f"Config file not found: {path}"
        raise ConfigurationError(msg)

    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in {path}: {exc}"
        raise ConfigurationError(msg) from exc

    if not isinstance(data, dict):
        msg = f"Expected a YAML mapping in {path}, got {type(data).__name__}"
        raise ConfigurationError(msg)

    try:
        return QueryConfig.model_validate(data)
    except Exception as exc:
        msg = f"Config validation failed for {path}: {exc}"
        raise ConfigurationError(msg) from exc
