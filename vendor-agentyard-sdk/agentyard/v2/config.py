"""Config loader — reads YAML mounted at /yard/config.yaml + env var overrides."""

import os
from pathlib import Path
from typing import Any

import yaml


class RuntimeConfig:
    """Per-node runtime configuration injected by the topology compiler."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str = "/yard/config.yaml") -> "RuntimeConfig":
        """Load config from mounted YAML, with env var overrides."""
        data: dict[str, Any] = {}
        config_path = Path(os.environ.get("YARD_CONFIG_PATH", path))
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

        # Env var overrides — flat keys for runtime tweaks
        env_overrides = {
            "system_id": os.environ.get("YARD_SYSTEM_ID"),
            "node_id": os.environ.get("YARD_NODE_ID"),
            "input_mode": os.environ.get("YARD_INPUT_MODE"),
            "output_mode": os.environ.get("YARD_OUTPUT_MODE"),
            "log_level": os.environ.get("LOG_LEVEL"),
        }
        for k, v in env_overrides.items():
            if v is not None:
                data[k] = v

        return cls(data)

    @property
    def system_id(self) -> str:
        return self._data.get("system_id", "")

    @property
    def node_id(self) -> str:
        return self._data.get("node_id", "")

    @property
    def input_mode(self) -> str:
        """One of: http, stream_consume, subscribe, aggregate, queue."""
        return self._data.get("input_mode", "http")

    @property
    def output_mode(self) -> str:
        """One of: sync, stream, emit, callback, queue."""
        return self._data.get("output_mode", "sync")

    @property
    def memory_contract(self) -> dict[str, Any]:
        """Memory ACL: {"reads": [...], "writes": [...], "scope": "..."}."""
        return self._data.get("memory", {"reads": [], "writes": [], "scope": "system"})

    @property
    def resources(self) -> dict[str, dict]:
        """Resource bindings: {"postgres": {"url": "..."}, "llm": {...}}."""
        return self._data.get("resources", {})

    @property
    def secrets(self) -> dict[str, str]:
        """Decrypted secrets injected at deploy time."""
        return self._data.get("secrets", {})

    @property
    def failure_policy(self) -> dict[str, Any]:
        return self._data.get("failure", {"mode": "retry", "max_retries": 2})

    @property
    def emit_targets(self) -> list[str]:
        """Topics/streams to emit output to (for output_mode=emit)."""
        return self._data.get("emit_targets", [])

    @property
    def subscribe_topics(self) -> list[str]:
        """Topics to subscribe to (for input_mode=subscribe)."""
        return self._data.get("subscribe_topics", [])

    @property
    def downstream_url(self) -> str:
        """For output_mode=stream — URL of next agent."""
        return self._data.get("downstream_url", "")

    @property
    def upstream_url(self) -> str:
        """For input_mode=stream_consume — URL of upstream agent."""
        return self._data.get("upstream_url", "")

    @property
    def redis_url(self) -> str:
        """Redis URL for stream/pubsub transports."""
        return (
            self._data.get("redis_url")
            or os.environ.get("YARD_REDIS_URL")
            or os.environ.get("REDIS_URL")
            or "redis://redis:6379/0"
        )

    @property
    def input_stream(self) -> str:
        """Stream name for input_mode=stream_consume."""
        return self._data.get("input_stream") or os.environ.get("YARD_INPUT_STREAM", "")

    @property
    def input_group(self) -> str:
        """Consumer group for input_mode=stream_consume."""
        return (
            self._data.get("input_group")
            or os.environ.get("YARD_INPUT_GROUP")
            or self.node_id
            or "yard"
        )

    @property
    def callback_url(self) -> str:
        """Callback URL for output_mode=callback."""
        return self._data.get("callback_url") or os.environ.get("YARD_CALLBACK_URL", "")

    @property
    def audit_stream(self) -> str:
        """Hybrid mode: Redis stream name the SDK mirrors each result to, in
        parallel with the sync response. Empty when off."""
        return self._data.get("audit_stream") or os.environ.get("YARD_AUDIT_STREAM", "")

    @property
    def transport_mode(self) -> str:
        """centralized | mesh | hybrid — surfaced purely for telemetry / logs."""
        return self._data.get("transport_mode") or os.environ.get("YARD_TRANSPORT_MODE", "centralized")

    @property
    def aggregate_batch_size(self) -> int:
        return int(self._data.get("aggregate_batch_size") or os.environ.get("YARD_AGG_BATCH", 5))

    @property
    def aggregate_window_seconds(self) -> float:
        return float(self._data.get("aggregate_window_seconds") or os.environ.get("YARD_AGG_WINDOW", 5.0))

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
