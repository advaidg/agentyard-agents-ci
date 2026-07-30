"""Distributed tracing with spans for agent invocations.

Provides lightweight tracing that propagates trace context across agent
boundaries. Completed spans are:

1. Published to a Redis stream for AgentYard's internal mission-control
   monitoring (``yard:traces:{agent_name}``).
2. Optionally batched and exported to an OTLP HTTP collector (e.g. Jaeger)
   when ``YARD_OTLP_ENDPOINT`` is set, so users can see real distributed
   traces in standard observability tooling.

Both exporters run independently — Redis is always attempted so the
platform never loses its internal view, and OTLP is layered on top for
external observability.

Usage:
    tracer = Tracer("my-agent", redis_url="redis://localhost:6379/0")

    with tracer.span("process_input") as span:
        span.set_attribute("input_size", 42)
        result = do_work()
        span.add_event("processed", items=len(result))

    # Or manually:
    span = tracer.start_span("manual_op")
    try:
        do_work()
        span.finish("ok")
    except Exception:
        span.finish("error")
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generator

import httpx
import redis.asyncio as aioredis

logger = logging.getLogger("agentyard.tracing")

_current_span: ContextVar["Span | None"] = ContextVar("current_span", default=None)

# Batching thresholds for OTLP export.
OTLP_BATCH_MAX_SPANS = 10
OTLP_BATCH_MAX_SECONDS = 5.0
OTLP_HTTP_TIMEOUT_SECONDS = 5.0

# Service version reported on the OTLP resource.
SDK_SERVICE_VERSION = "0.4.0"


@dataclass
class Span:
    """A single span in a distributed trace."""

    trace_id: str
    span_id: str
    parent_id: str | None
    operation: str
    agent_name: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    status: str = "running"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    # Wall-clock start time in nanoseconds since epoch. Captured on
    # construction so OTLP exporters can report absolute timestamps
    # instead of the monotonic clock used for duration math.
    start_time_unix_nano: int = field(
        default_factory=lambda: int(time.time() * 1_000_000_000)
    )
    end_time_unix_nano: int | None = None

    @property
    def duration_ms(self) -> int:
        """Elapsed time in milliseconds. Returns 0 if span is still running."""
        if self.end_time is None:
            return int((time.monotonic() - self.start_time) * 1000)
        return int((self.end_time - self.start_time) * 1000)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a key-value attribute to this span."""
        self.attributes = {**self.attributes, key: value}

    def add_event(self, name: str, **attrs: Any) -> None:
        """Record a timestamped event within this span."""
        event = {
            "name": name,
            "timestamp": time.monotonic(),
            **attrs,
        }
        self.events = [*self.events, event]

    def finish(self, status: str = "ok") -> None:
        """Mark the span as complete with a final status."""
        self.end_time = time.monotonic()
        self.status = status
        # Derive the wall-clock end time from the monotonic delta to
        # keep duration and end-time consistent across clock skew.
        delta_ns = int((self.end_time - self.start_time) * 1_000_000_000)
        self.end_time_unix_nano = self.start_time_unix_nano + delta_ns

    def to_dict(self) -> dict[str, Any]:
        """Serialize the span for export."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "agent_name": self.agent_name,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": [
                {k: v for k, v in e.items() if k != "timestamp"}
                for e in self.events
            ],
        }


def _to_otlp_trace_id(trace_id: str) -> str:
    """Normalise a trace id to 32 hex chars (16 bytes) for OTLP.

    AgentYard generates trace ids from ``uuid.uuid4().hex`` (32 hex
    chars), but users may pass in UUID strings with dashes. We strip
    dashes and pad/truncate to exactly 32 chars.
    """
    cleaned = trace_id.replace("-", "")
    if len(cleaned) >= 32:
        return cleaned[:32]
    return cleaned.rjust(32, "0")


def _to_otlp_span_id(span_id: str | None) -> str:
    """Normalise a span id to 16 hex chars (8 bytes) for OTLP.

    Empty or ``None`` parent ids are rendered as an empty string, which
    OTLP interprets as "no parent" (i.e. root span).
    """
    if not span_id:
        return ""
    cleaned = span_id.replace("-", "")
    if len(cleaned) >= 16:
        return cleaned[:16]
    return cleaned.rjust(16, "0")


def _otlp_attribute(key: str, value: Any) -> dict[str, Any]:
    """Convert a Python value into an OTLP attribute entry."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _otlp_status_code(status: str) -> int:
    """Map AgentYard span status to OTLP status code.

    0 = UNSET, 1 = OK, 2 = ERROR.
    """
    if status == "ok":
        return 1
    if status == "error":
        return 2
    return 0


def _span_to_otlp(span: Span) -> dict[str, Any]:
    """Convert a single Span to the OTLP JSON span shape."""
    end_unix_nano = (
        span.end_time_unix_nano
        if span.end_time_unix_nano is not None
        else int(time.time() * 1_000_000_000)
    )
    attributes = [
        _otlp_attribute(k, v) for k, v in span.attributes.items()
    ]
    attributes.append(_otlp_attribute("agentyard.agent_name", span.agent_name))
    return {
        "traceId": _to_otlp_trace_id(span.trace_id),
        "spanId": _to_otlp_span_id(span.span_id),
        "parentSpanId": _to_otlp_span_id(span.parent_id),
        "name": span.operation,
        "startTimeUnixNano": span.start_time_unix_nano,
        "endTimeUnixNano": end_unix_nano,
        # kind=1 == INTERNAL. Agent-to-agent calls are logically
        # internal to the AgentYard execution graph.
        "kind": 1,
        "status": {"code": _otlp_status_code(span.status)},
        "attributes": attributes,
    }


def _build_otlp_payload(
    spans: list[Span], agent_name: str
) -> dict[str, Any]:
    """Build an OTLP/HTTP ``/v1/traces`` JSON payload."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _otlp_attribute(
                            "service.name", f"agentyard-{agent_name}"
                        ),
                        _otlp_attribute(
                            "service.version", SDK_SERVICE_VERSION
                        ),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "agentyard-sdk"},
                        "spans": [_span_to_otlp(s) for s in spans],
                    }
                ],
            }
        ]
    }


class Tracer:
    """Lightweight distributed tracer for agent invocations.

    Creates spans that form a trace tree. Completed spans are exported
    to Redis streams for AgentYard's internal monitoring, and optionally
    batched and sent to an OTLP HTTP collector (e.g. Jaeger) when
    ``YARD_OTLP_ENDPOINT`` is set.
    """

    def __init__(
        self,
        agent_name: str,
        redis_url: str = "",
        otlp_endpoint: str = "",
    ) -> None:
        self.agent_name = agent_name
        self._redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")
        self._otlp_endpoint = otlp_endpoint or os.environ.get(
            "YARD_OTLP_ENDPOINT", ""
        )
        self._redis: aioredis.Redis | None = None
        self._http: httpx.AsyncClient | None = None

        # OTLP batching state. Protected by ``_otlp_lock`` because
        # ``export`` may be called concurrently from multiple tasks.
        self._otlp_buffer: list[Span] = []
        self._otlp_buffer_started_at: float | None = None
        self._otlp_lock = asyncio.Lock()
        self._otlp_flush_task: asyncio.Task[None] | None = None

    async def _get_redis(self) -> aioredis.Redis | None:
        if not self._redis and self._redis_url:
            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._redis

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=OTLP_HTTP_TIMEOUT_SECONDS
            )
        return self._http

    def start_span(
        self, operation: str, parent: "Span | None" = None
    ) -> Span:
        """Start a new span. Inherits trace_id from parent if provided."""
        parent = parent or _current_span.get()
        trace_id = parent.trace_id if parent else uuid.uuid4().hex
        parent_id = parent.span_id if parent else None

        span = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent_id,
            operation=operation,
            agent_name=self.agent_name,
        )
        _current_span.set(span)
        return span

    @contextmanager
    def span(self, operation: str) -> Generator["Span", None, None]:
        """Context manager for automatic span lifecycle.

        Starts a span, yields it for attribute/event recording,
        and finishes it when the block exits. Sets status to "error"
        if an exception propagates.
        """
        previous = _current_span.get()
        new_span = self.start_span(operation, parent=previous)
        try:
            yield new_span
            new_span.finish("ok")
        except Exception:
            new_span.finish("error")
            raise
        finally:
            _current_span.set(previous)

    async def export(self, span: Span) -> None:
        """Export a completed span.

        Always attempts to publish to the Redis stream so AgentYard's
        mission control keeps working. If ``YARD_OTLP_ENDPOINT`` is set,
        the span is also queued for batched OTLP export so Jaeger (or
        any OTLP-compatible collector) sees the trace.
        """
        await self._export_redis(span)
        if self._otlp_endpoint:
            await self._queue_otlp(span)

    async def _export_redis(self, span: Span) -> None:
        """Push a span to the Redis trace stream (best-effort)."""
        r = await self._get_redis()
        if not r:
            return
        try:
            stream_key = f"yard:traces:{self.agent_name}"
            await r.xadd(
                stream_key,
                {"span": json.dumps(span.to_dict())},
                maxlen=5000,
            )
        except Exception as exc:
            logger.warning("redis_trace_export_failed: %s", exc)

    async def _queue_otlp(self, span: Span) -> None:
        """Add a span to the OTLP batch buffer.

        Flushes synchronously when the buffer hits
        ``OTLP_BATCH_MAX_SPANS``. Otherwise schedules a deferred flush
        so partial batches don't linger longer than
        ``OTLP_BATCH_MAX_SECONDS``.
        """
        async with self._otlp_lock:
            if not self._otlp_buffer:
                self._otlp_buffer_started_at = time.monotonic()
            self._otlp_buffer = [*self._otlp_buffer, span]
            should_flush_now = (
                len(self._otlp_buffer) >= OTLP_BATCH_MAX_SPANS
            )

        if should_flush_now:
            await self._flush_otlp()
            return

        # Schedule a background flush if one isn't already pending.
        if (
            self._otlp_flush_task is None
            or self._otlp_flush_task.done()
        ):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._otlp_flush_task = loop.create_task(
                self._deferred_flush()
            )

    async def _deferred_flush(self) -> None:
        """Wait until the batch age hits the max, then flush."""
        try:
            await asyncio.sleep(OTLP_BATCH_MAX_SECONDS)
            await self._flush_otlp()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("otlp_deferred_flush_failed: %s", exc)

    async def _flush_otlp(self) -> None:
        """Drain the OTLP buffer and POST it to the collector."""
        async with self._otlp_lock:
            if not self._otlp_buffer:
                return
            to_send = self._otlp_buffer
            self._otlp_buffer = []
            self._otlp_buffer_started_at = None

        if not self._otlp_endpoint:
            return

        payload = _build_otlp_payload(to_send, self.agent_name)
        try:
            client = await self._get_http()
            await client.post(
                self._otlp_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            logger.warning("otlp_flush_failed endpoint=%s: %s", self._otlp_endpoint, exc)

    async def flush(self) -> None:
        """Force an immediate flush of any buffered OTLP spans."""
        if self._otlp_endpoint:
            await self._flush_otlp()

    async def close(self) -> None:
        """Flush buffers and close underlying clients."""
        if self._otlp_flush_task and not self._otlp_flush_task.done():
            self._otlp_flush_task.cancel()
            try:
                await self._otlp_flush_task
            except (asyncio.CancelledError, Exception):
                pass
        await self.flush()
        if self._redis:
            await self._redis.close()
            self._redis = None
        if self._http:
            await self._http.aclose()
            self._http = None

    @staticmethod
    def current() -> "Span | None":
        """Get the current active span from context."""
        return _current_span.get()
