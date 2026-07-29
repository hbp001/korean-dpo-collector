"""
Config loader — reads config/config.yaml and exposes a typed dict.
Usage:
    from config.config_loader import get_config
    cfg = get_config()
    model_name = cfg["models"]["vllm"]["name"]
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG: Dict[str, Any] = {}
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def get_config(config_path: str | None = None) -> Dict[str, Any]:
    """Return the global config dict, loading from disk on first call."""
    global _CONFIG, _CONFIG_PATH
    if config_path:
        _CONFIG_PATH = Path(config_path)
    if not _CONFIG:
        _CONFIG = _load(_CONFIG_PATH)
    return _CONFIG


def reload_config(config_path: str | None = None) -> Dict[str, Any]:
    """Force-reload config from disk (useful for tests)."""
    global _CONFIG
    _CONFIG = {}
    return get_config(config_path)


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve environment variable overrides  (e.g. NEO4J_PASSWORD)
    _resolve_env(cfg)
    return cfg


def _resolve_env(obj: Any) -> None:
    """Walk the config tree and replace ${ENV_VAR} placeholders."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_var = v[2:-1]
                obj[k] = os.environ.get(env_var, v)
            else:
                _resolve_env(v)
    elif isinstance(obj, list):
        for item in obj:
            _resolve_env(item)
