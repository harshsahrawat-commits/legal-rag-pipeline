"""Configuration loading for the hallucination mitigation module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.hallucination._exceptions import HallucinationConfigError
from src.hallucination._models import HallucinationConfig
from src.utils._logging import get_logger

_log = get_logger(__name__)

_DEFAULT_CONFIG_PATH = Path("configs/hallucination.yaml")


def load_hallucination_config(
    config_path: Path | None = None,
) -> HallucinationConfig:
    """Load hallucination config from a YAML file.

    Args:
        config_path: Path to YAML config. Defaults to configs/hallucination.yaml.

    Returns:
        Parsed HallucinationConfig.

    Raises:
        HallucinationConfigError: If the file cannot be read or parsed.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    if not path.exists():
        _log.info("hallucination_config_not_found", path=str(path))
        return HallucinationConfig()

    try:
        import yaml

        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return HallucinationConfig(**raw)
    except ImportError:
        _log.info("pyyaml_not_installed_using_defaults")
        return HallucinationConfig()
    except Exception as exc:
        msg = f"Failed to load hallucination config from {path}: {exc}"
        raise HallucinationConfigError(msg) from exc
