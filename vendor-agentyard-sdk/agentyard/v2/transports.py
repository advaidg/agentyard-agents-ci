"""Adaptive transports for AgentYard v2.

Decouples HOW an agent receives inputs and emits outputs from WHAT it does.
The same handler runs unchanged whether it's invoked synchronously over HTTP,
consumes from a Redis stream, or subscribes to a pub/sub topic.

Input modes:
    http             — FastAPI POST / triggers the handler (default)
    stream_consume   — Background loop reads from a Redis stream (XREADGROUP)
    subscribe        — Background loop listens to a Redis pub/sub channel
    aggregate        — Buffer N inputs then trigger handler with the batch

Output modes:
    sync       — Return the result in the HTTP response (default)
    stream     — POST result to downstream_url and return ack
    emit       — Publish to one or more Redis streams/channels
    callback   — POST result to a callback URL, return ack immediately
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable

import httpx
import redis.asyncio as redis_async

logger = logging.getLogger("yard.transport")

HandlerFn = Callable[[Any], Awaitable[Any]]
# Dispatcher signature: (payload, invocation_id, trace_id) -> None
DispatchFn = Callable[[Any, str, str], Awaitable[None]]


# ---------- Output Transports ----------

class OutputTransport:
    """Base class — turns a handler result into a delivery action.

    All transports accept ``trace_id`` so distributed tracing survives
    peer-to-peer forwarding, Redis stream hops, and pub/sub handoffs.
    Consumers that pull from streams can recover the trace context and
    continue the parent span.
    """

    async def deliver(self, result: Any, *, invocation_id: str, trace_id: str = "") -> dict:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class SyncOutput(OutputTransport):
    """Return the result directly to the caller."""

    async def deliver(self, result: Any, *, invocation_id: str, trace_id: str = "") -> dict:
        return {"mode": "sync", "result": result}


class StreamOutput(OutputTransport):
    """Forward the result to the next agent in the chain."""

    def __init__(self, downstream_url: str, *, timeout: float = 30.0):
        if not downstream_url:
            raise ValueError("StreamOutput requires downstream_url")
        self._url = downstream_url.rstrip("/")
        # Connection pooling — one client per node lifetime, reused for every
        # forwarded call. Cuts per-hop latency in mesh chains.
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )

    async def deliver(self, result: Any, *, invocation_id: str, trace_id: str = "") -> dict:
        headers: dict[str, str] = {}
        if invocation_id:
            headers["X-Invocation-ID"] = invocation_id
        if trace_id:
            # Standard W3C-ish propagation header — consumers can pick it up
            # without needing a vendor-specific SDK.
            headers["X-Trace-ID"] = trace_id
        try:
            resp = await self._client.post(
                f"{self._url}/invoke",
                json={"input": result},
                headers=headers,
            )
            resp.raise_for_status()
            return {"mode": "stream", "downstream": self._url, "status": "forwarded"}
        except httpx.HTTPError as exc:
            logger.warning("stream_forward_failed url=%s err=%s", self._url, exc)
            return {"mode": "stream", "downstream": self._url, "status": "failed", "error": str(exc)}

    async def close(self) -> None:
        await self._client.aclose()


class EmitOutput(OutputTransport):
    """Publish the result to one or more Redis streams/channels.

    Every emitted payload carries ``trace_id`` + ``invocation_id`` so the
    consumer's SDK can continue the distributed trace without needing an
    out-of-band context propagation channel.
    """

    def __init__(self, redis_url: str, targets: list[str], *, maxlen: int = 10000):
        if not targets:
            raise ValueError("EmitOutput requires at least one target")
        self._redis = redis_async.from_url(redis_url, decode_responses=True)
        self._targets = targets
        self._maxlen = maxlen

    async def deliver(self, result: Any, *, invocation_id: str, trace_id: str = "") -> dict:
        payload = json.dumps({
            "invocation_id": invocation_id,
            "trace_id": trace_id,
            "ts": time.time(),
            "result": result,
        }, default=str)
        delivered: list[str] = []
        for target in self._targets:
            try:
                if target.startswith("stream:"):
                    # Approximate MAXLEN trim — stops Redis from growing unbounded
                    # over a long-running mesh topology.
                    await self._redis.xadd(
                        target[len("stream:"):],
                        {"data": payload},
                        maxlen=self._maxlen,
                        approximate=True,
                    )
                elif target.startswith("channel:"):
                    await self._redis.publish(target[len("channel:"):], payload)
                else:  # default to stream semantics
                    await self._redis.xadd(
                        target, {"data": payload}, maxlen=self._maxlen, approximate=True,
                    )
                delivered.append(target)
            except Exception as exc:
                logger.warning("emit_failed target=%s err=%s", target, exc)
        return {"mode": "emit", "targets": delivered, "count": len(delivered)}

    async def close(self) -> None:
        await self._redis.aclose()


class CallbackOutput(OutputTransport):
    """Fire-and-forget POST to a callback URL."""

    def __init__(self, callback_url: str, *, timeout: float = 10.0):
        if not callback_url:
            raise ValueError("CallbackOutput requires callback_url")
        self._url = callback_url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def deliver(self, result: Any, *, invocation_id: str, trace_id: str = "") -> dict:
        async def _post() -> None:
            try:
                await self._client.post(
                    self._url,
                    json={"invocation_id": invocation_id, "trace_id": trace_id, "result": result},
                )
            except Exception as exc:
                logger.warning("callback_failed url=%s err=%s", self._url, exc)

        # Schedule without awaiting — caller gets immediate ack
        asyncio.create_task(_post())
        return {"mode": "callback", "url": self._url, "status": "scheduled"}

    async def close(self) -> None:
        await self._client.aclose()


def build_output_transport(mode: str, *, redis_url: str, downstream_url: str,
                            emit_targets: list[str], callback_url: str) -> OutputTransport:
    """Factory — picks the right output transport from declared mode."""
    if mode == "stream":
        return StreamOutput(downstream_url)
    if mode == "emit":
        return EmitOutput(redis_url, emit_targets)
    if mode == "callback":
        return CallbackOutput(callback_url)
    return SyncOutput()


# ---------- Input Transports ----------

class InputTransport:
    """Base class — drives handler invocation from a non-HTTP source."""

    async def start(self, dispatch: DispatchFn) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        pass


class HttpInput(InputTransport):
    """No-op — HTTP server is started by uvicorn separately."""

    async def start(self, dispatch: DispatchFn) -> None:
        return

    async def stop(self) -> None:
        return


class StreamConsumeInput(InputTransport):
    """Read events from a Redis stream as a consumer group member."""

    def __init__(self, redis_url: str, stream: str, *, group: str = "yard", consumer: str | None = None,
                 block_ms: int = 5000, batch: int = 1):
        if not stream:
            raise ValueError("StreamConsumeInput requires stream name")
        self._redis = redis_async.from_url(redis_url, decode_responses=True)
        self._stream = stream
        self._group = group
        self._consumer = consumer or f"yard-{uuid.uuid4().hex[:8]}"
        self._block_ms = block_ms
        self._batch = batch
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    async def _ensure_group(self) -> None:
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except Exception as exc:
            # BUSYGROUP — group exists, that's fine
            if "BUSYGROUP" not in str(exc):
                logger.warning("xgroup_create_failed stream=%s err=%s", self._stream, exc)

    async def _loop(self, dispatch: DispatchFn) -> None:
        await self._ensure_group()
        while not self._stopped.is_set():
            try:
                resp = await self._redis.xreadgroup(
                    self._group, self._consumer,
                    {self._stream: ">"}, count=self._batch, block=self._block_ms,
                )
            except Exception as exc:
                # If the stream/group disappeared (e.g. external delete), recreate and retry.
                if "NOGROUP" in str(exc):
                    logger.info("xreadgroup_nogroup_recreating stream=%s group=%s", self._stream, self._group)
                    await self._ensure_group()
                    continue
                logger.warning("xreadgroup_failed stream=%s err=%s", self._stream, exc)
                await asyncio.sleep(1.0)
                continue

            if not resp:
                continue

            for _stream_name, entries in resp:
                for entry_id, fields in entries:
                    raw = fields.get("data") or fields.get("input") or json.dumps(fields)
                    try:
                        payload = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        payload = {"data": raw}
                    # Lift invocation_id + trace_id back out of the envelope so
                    # the handler's ctx.tracer can continue the parent span.
                    invocation_id = ""
                    trace_id = ""
                    inner = payload
                    if isinstance(payload, dict):
                        invocation_id = str(payload.get("invocation_id") or "")
                        trace_id = str(payload.get("trace_id") or "")
                        # Strip the envelope so the handler sees only the result.
                        if "result" in payload:
                            inner = payload["result"]
                    if not invocation_id:
                        invocation_id = fields.get("invocation_id") or entry_id
                    try:
                        await dispatch(inner, invocation_id, trace_id)
                    except Exception as exc:
                        logger.exception("stream_handler_failed entry=%s err=%s", entry_id, exc)
                    finally:
                        try:
                            await self._redis.xack(self._stream, self._group, entry_id)
                        except Exception:
                            pass

    async def start(self, dispatch: DispatchFn) -> None:
        logger.info("stream_consumer_starting stream=%s group=%s consumer=%s",
                    self._stream, self._group, self._consumer)
        self._task = asyncio.create_task(self._loop(dispatch))

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self._redis.aclose()


class SubscribeInput(InputTransport):
    """Listen on Redis pub/sub channels and dispatch each message."""

    def __init__(self, redis_url: str, channels: list[str]):
        if not channels:
            raise ValueError("SubscribeInput requires at least one channel")
        self._redis = redis_async.from_url(redis_url, decode_responses=True)
        self._channels = channels
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    async def _loop(self, dispatch: DispatchFn) -> None:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(*self._channels)
        logger.info("subscribed channels=%s", self._channels)
        try:
            while not self._stopped.is_set():
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    continue
                raw = msg.get("data")
                try:
                    payload = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    payload = {"data": raw}
                invocation_id = ""
                trace_id = ""
                inner = payload
                if isinstance(payload, dict):
                    invocation_id = str(payload.get("invocation_id") or "")
                    trace_id = str(payload.get("trace_id") or "")
                    if "result" in payload:
                        inner = payload["result"]
                if not invocation_id:
                    invocation_id = f"sub-{uuid.uuid4().hex[:8]}"
                try:
                    await dispatch(inner, invocation_id, trace_id)
                except Exception as exc:
                    logger.exception("subscribe_handler_failed err=%s", exc)
        finally:
            await pubsub.unsubscribe(*self._channels)
            await pubsub.aclose()

    async def start(self, dispatch: DispatchFn) -> None:
        self._task = asyncio.create_task(self._loop(dispatch))

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self._redis.aclose()


class AggregateInput(InputTransport):
    """Buffer HTTP-delivered inputs and dispatch when batch is full or window elapses.

    The HTTP endpoint still accepts POSTs, but instead of dispatching one-at-a-time
    it accumulates them. Used for fanout-merge or map-reduce final stages.
    """

    def __init__(self, batch_size: int = 5, window_seconds: float = 5.0):
        self._batch_size = max(1, batch_size)
        self._window = window_seconds
        # Each entry is (payload, invocation_id, trace_id)
        self._buffer: list[tuple[Any, str, str]] = []
        self._lock = asyncio.Lock()
        self._dispatch: DispatchFn | None = None
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    async def offer(self, payload: Any, invocation_id: str, trace_id: str = "") -> dict:
        """Called by the HTTP layer for each incoming POST."""
        async with self._lock:
            self._buffer.append((payload, invocation_id, trace_id))
            ready = len(self._buffer) >= self._batch_size
            batch = self._buffer.copy() if ready else []
            if ready:
                self._buffer.clear()
        if ready and self._dispatch:
            payloads = [p for p, _, _ in batch]
            # Prefer the first non-empty trace_id so the batch inherits a
            # lineage rather than starting a fresh trace.
            batch_trace = next((t for _, _, t in batch if t), "")
            agg_id = f"agg-{uuid.uuid4().hex[:8]}"
            await self._dispatch(payloads, agg_id, batch_trace)
            return {"buffered": False, "dispatched": True, "size": len(batch)}
        return {"buffered": True, "size_pending": len(self._buffer)}

    async def _flusher(self) -> None:
        while not self._stopped.is_set():
            await asyncio.sleep(self._window)
            async with self._lock:
                if not self._buffer:
                    continue
                batch = self._buffer.copy()
                self._buffer.clear()
            if self._dispatch:
                payloads = [p for p, _, _ in batch]
                batch_trace = next((t for _, _, t in batch if t), "")
                agg_id = f"agg-{uuid.uuid4().hex[:8]}"
                try:
                    await self._dispatch(payloads, agg_id, batch_trace)
                except Exception as exc:
                    logger.exception("aggregate_flush_failed err=%s", exc)

    async def start(self, dispatch: DispatchFn) -> None:
        self._dispatch = dispatch
        self._task = asyncio.create_task(self._flusher())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass


def build_input_transport(mode: str, *, redis_url: str, stream: str, group: str,
                           channels: list[str], batch_size: int, window_seconds: float
                           ) -> InputTransport:
    """Factory — picks input transport from declared mode."""
    if mode == "stream_consume":
        return StreamConsumeInput(redis_url, stream, group=group)
    if mode == "subscribe":
        return SubscribeInput(redis_url, channels)
    if mode == "aggregate":
        return AggregateInput(batch_size=batch_size, window_seconds=window_seconds)
    return HttpInput()
