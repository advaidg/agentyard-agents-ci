"""SDK configuration — persisted to ~/.agentyard/config.json."""

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".agentyard"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS = {
    "registry_url": "http://localhost:8000",
    "token": "",
}


def _ensure_config_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    _ensure_config_dir()
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULTS, **json.load(f)}
    return dict(DEFAULTS)


def save_config(config: dict) -> None:
    _ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_value(key: str) -> str:
    config = load_config()
    return config.get(key, DEFAULTS.get(key, ""))


def set_value(key: str, value: str) -> None:
    config = load_config()
    config[key] = value
    save_config(config)
