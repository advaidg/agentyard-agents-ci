"""Agent-side metrics — reports to Redis for AgentYard analytics
and exposes Prometheus counters/histograms for scraping.

Supports four metric types:
- **Invocations**: Built-in call/success/error tracking.
- **Counters**: Monotonically increasing values with optional labels.
- **Gauges**: Point-in-time values (queue depth, memory, etc.).
- **Histograms**: Distributions stored as capped lists in Redis.

Plus LLM-specific series used by the ``AgentYard LLM Agents`` Grafana
dashboard:

- ``agentyard_agent_invocations_total{agent, status, model}``
- ``agentyard_agent_duration_seconds{agent, model}`` (histogram)
- ``agentyard_tokens_in_total{agent, model}``
- ``agentyard_tokens_out_total{agent, model}``
- ``agentyard_cost_usd_total{agent, model}``
- ``agentyard_tool_calls_total{agent, tool, status}``
- ``agentyard_tool_call_duration_seconds{agent, tool}`` (histogram)

All Redis keys live under ``yard:metrics:{agent_name}:*`` and expire
after 24 hours by default. Prometheus series live in the default
registry and are exposed via the agent's ``/metrics`` endpoint.
"""

from __future__ import annotations

import os
from typing import Any

import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram

_METRIC_TTL = 86400  # 24 hours
_HISTOGRAM_MAX_LEN = 1000

# Shared histogram bucket layout — covers sub-second tool calls through
# multi-minute LLM generations without blowing label cardinality.
_DURATION_BUCKETS = (
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)

# Prometheus collectors are module-level so multiple MetricsReporter
# instances in the same process share a single registry entry. The
# prometheus_client library raises if we register the same metric twice,
# so guard against double-import (e.g. under ``python -m`` + pytest).

def _get_or_create_counter(
    name: str, documentation: str, labelnames: tuple[str, ...]
) -> Counter:
    try:
        return Counter(name, documentation, labelnames)
    except ValueError:  # Already registered
        from prometheus_client import REGISTRY

        collector = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        if collector is None:
            raise
        return collector  # type: ignore[return-value]


def _get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: tuple[str, ...],
    buckets: tuple[float, ...] = _DURATION_BUCKETS,
) -> Histogram:
    try:
        return Histogram(name, documentation, labelnames, buckets=buckets)
    except ValueError:
        from prometheus_client import REGISTRY

        collector = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
        if collector is None:
            raise
        return collector  # type: ignore[return-value]


AGENT_INVOCATIONS = _get_or_create_counter(
    "agentyard_agent_invocations_total",
    "Total agent invocations by status and model.",
    ("agent", "status", "model"),
)

AGENT_DURATION = _get_or_create_histogram(
    "agentyard_agent_duration_seconds",
    "Agent invocation wall-clock duration in seconds.",
    ("agent", "model"),
)

TOKENS_IN = _get_or_create_counter(
    "agentyard_tokens_in_total",
    "Input (prompt) tokens consumed by an agent.",
    ("agent", "model"),
)

TOKENS_OUT = _get_or_create_counter(
    "agentyard_tokens_out_total",
    "Output (completion) tokens produced by an agent.",
    ("agent", "model"),
)

COST_USD = _get_or_create_counter(
    "agentyard_cost_usd_total",
    "USD cost incurred by an agent, reported by the agent itself.",
    ("agent", "model"),
)

TOOL_CALLS = _get_or_create_counter(
    "agentyard_tool_calls_total",
    "Total tool invocations by agent, tool, and status.",
    ("agent", "tool", "status"),
)

TOOL_CALL_DURATION = _get_or_create_histogram(
    "agentyard_tool_call_duration_seconds",
    "Tool invocation wall-clock duration in seconds.",
    ("agent", "tool"),
)

_UNKNOWN_MODEL = "unknown"


class MetricsReporter:
    """Reports agent metrics to Redis AND to Prometheus.

    The Redis writes feed platform-side analytics and the Mission
    Control UI, while the Prometheus writes feed the LLM dashboard via
    the agent's ``/metrics`` endpoint.
    """

    def __init__(self, agent_name: str, redis_url: str = "") -> None:
        self.agent_name = agent_name
        self._redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis | None:
        if not self._redis and self._redis_url:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _base_key(self) -> str:
        return f"yard:metrics:{self.agent_name}"

    @staticmethod
    def _label_suffix(labels: dict[str, str] | None) -> str:
        """Encode labels into a deterministic key suffix."""
        if not labels:
            return ""
        sorted_pairs = sorted(labels.items())
        return ":" + ",".join(f"{k}={v}" for k, v in sorted_pairs)

    async def record_invocation(
        self,
        duration_ms: int,
        success: bool,
        *,
        model: str | None = None,
        status: str | None = None,
    ) -> None:
        """Record an invocation — writes to both Redis and Prometheus.

        ``status`` overrides the success/error mapping; useful when callers
        want a finer-grained label (e.g. ``"timeout"``, ``"invalid_input"``).
        """
        resolved_status = status or ("success" if success else "error")
        resolved_model = model or _UNKNOWN_MODEL

        # Prometheus export — never fails.
        try:
            AGENT_INVOCATIONS.labels(
                agent=self.agent_name,
                status=resolved_status,
                model=resolved_model,
            ).inc()
            AGENT_DURATION.labels(
                agent=self.agent_name, model=resolved_model
            ).observe(duration_ms / 1000.0)
        except Exception:
            pass

        # Redis — may fail silently when Redis is unavailable.
        r = await self._get_redis()
        if not r:
            return
        key = f"yard:agent:metrics:{self.agent_name}"
        try:
            pipe = r.pipeline()
            pipe.hincrby(key, "total_calls", 1)
            pipe.hincrby(key, "success" if success else "error", 1)
            pipe.hincrby(key, "total_ms", duration_ms)
            pipe.lpush(f"{key}:latencies", duration_ms)
            pipe.ltrim(f"{key}:latencies", 0, _HISTOGRAM_MAX_LEN - 1)
            pipe.expire(key, _METRIC_TTL)
            pipe.expire(f"{key}:latencies", _METRIC_TTL)
            await pipe.execute()
        except Exception:
            pass

    async def record_tokens(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Record input/output token counts for a single LLM call."""
        resolved_model = model or _UNKNOWN_MODEL
        try:
            if tokens_in:
                TOKENS_IN.labels(
                    agent=self.agent_name, model=resolved_model
                ).inc(float(tokens_in))
            if tokens_out:
                TOKENS_OUT.labels(
                    agent=self.agent_name, model=resolved_model
                ).inc(float(tokens_out))
        except Exception:
            pass

        r = await self._get_redis()
        if not r:
            return
        base = f"{self._base_key()}:tokens"
        try:
            pipe = r.pipeline()
            pipe.hincrby(f"{base}:in", resolved_model, int(tokens_in))
            pipe.hincrby(f"{base}:out", resolved_model, int(tokens_out))
            pipe.expire(f"{base}:in", _METRIC_TTL)
            pipe.expire(f"{base}:out", _METRIC_TTL)
            await pipe.execute()
        except Exception:
            pass

    async def record_cost(self, model: str, cost_usd: float) -> None:
        """Record USD cost for a single LLM call."""
        if cost_usd <= 0:
            return
        resolved_model = model or _UNKNOWN_MODEL
        try:
            COST_USD.labels(
                agent=self.agent_name, model=resolved_model
            ).inc(float(cost_usd))
        except Exception:
            pass

        r = await self._get_redis()
        if not r:
            return
        try:
            pipe = r.pipeline()
            pipe.hincrbyfloat(
                f"{self._base_key()}:cost_usd", resolved_model, float(cost_usd)
            )
            pipe.expire(f"{self._base_key()}:cost_usd", _METRIC_TTL)
            await pipe.execute()
        except Exception:
            pass

    async def record_tool_call(
        self,
        tool: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a tool invocation (MCP or built-in)."""
        try:
            TOOL_CALLS.labels(
                agent=self.agent_name, tool=tool, status=status
            ).inc()
            TOOL_CALL_DURATION.labels(
                agent=self.agent_name, tool=tool
            ).observe(max(0.0, duration_seconds))
        except Exception:
            pass

        r = await self._get_redis()
        if not r:
            return
        try:
            pipe = r.pipeline()
            pipe.hincrby(
                f"{self._base_key()}:tool_calls:{tool}", status, 1
            )
            pipe.expire(
                f"{self._base_key()}:tool_calls:{tool}", _METRIC_TTL
            )
            await pipe.execute()
        except Exception:
            pass

    async def increment_counter(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a named counter with optional labels.

        Counters are stored as Redis hashes so multiple label
        combinations live under one key.
        """
        r = await self._get_redis()
        if not r:
            return
        key = f"{self._base_key()}:counter:{name}"
        field = "value" + self._label_suffix(labels)
        try:
            pipe = r.pipeline()
            pipe.hincrby(key, field, value)
            pipe.expire(key, _METRIC_TTL)
            await pipe.execute()
        except Exception:
            pass

    async def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value (e.g., queue depth, memory usage).

        Gauges represent a point-in-time measurement and can go
        up or down freely.
        """
        r = await self._get_redis()
        if not r:
            return
        key = f"{self._base_key()}:gauge:{name}"
        field = "value" + self._label_suffix(labels)
        try:
            pipe = r.pipeline()
            pipe.hset(key, field, str(value))
            pipe.expire(key, _METRIC_TTL)
            await pipe.execute()
        except Exception:
            pass

    async def record_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a value in a histogram (e.g., response size, latency).

        Values are stored in a capped Redis list (most recent 1000).
        """
        r = await self._get_redis()
        if not r:
            return
        suffix = self._label_suffix(labels)
        key = f"{self._base_key()}:hist:{name}{suffix}"
        try:
            pipe = r.pipeline()
            pipe.rpush(key, str(value))
            pipe.ltrim(key, -_HISTOGRAM_MAX_LEN, -1)
            pipe.expire(key, _METRIC_TTL)
            await pipe.execute()
        except Exception:
            pass

    async def get_summary(self) -> dict[str, Any]:
        """Get all metrics for this agent as a summary dict.

        Returns counters, gauges, and histogram lengths keyed by name.
        """
        r = await self._get_redis()
        if not r:
            return {}

        summary: dict[str, Any] = {}

        # Invocation metrics
        inv_key = f"yard:agent:metrics:{self.agent_name}"
        try:
            inv_data = await r.hgetall(inv_key)
            if inv_data:
                summary["invocations"] = {
                    k: int(v) for k, v in inv_data.items()
                }
        except Exception:
            pass

        # Scan for custom metrics
        base = self._base_key()
        try:
            for metric_type in ("counter", "gauge", "hist"):
                pattern = f"{base}:{metric_type}:*"
                cursor = "0"
                type_data: dict[str, Any] = {}
                while True:
                    cursor, keys = await r.scan(
                        cursor=cursor, match=pattern, count=100
                    )
                    for key in keys:
                        metric_name = key.split(f":{metric_type}:", 1)[-1]
                        if metric_type in ("counter", "gauge"):
                            type_data[metric_name] = await r.hgetall(key)
                        else:
                            type_data[metric_name] = await r.llen(key)
                    if cursor == "0" or cursor == 0:
                        break
                if type_data:
                    summary[f"{metric_type}s"] = type_data
        except Exception:
            pass

        return summary

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
