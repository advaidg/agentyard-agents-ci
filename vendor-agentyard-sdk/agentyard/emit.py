"""Multi-destination output emission for AgentYard agents.

Agents can send outputs, progress updates, and structured logs to
multiple destinations simultaneously based on configuration.

Configuration via environment variables:
    YARD_EMIT_TARGETS   — comma-separated list of targets (default: ``http``)
    YARD_EMIT_CHANNEL   — Redis pub/sub channel template
                          (default: ``yard:events:{system_id}``)
    YARD_EMIT_STREAM    — Redis stream name template
                          (default: ``yard:system:{system_id}:events``)
    YARD_EMIT_CALLBACK  — default callback URL for the ``callback`` target
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("agentyard.emit")

if TYPE_CHECKING:
    from agentyard.context import YardContext


class EmitTarget(Enum):
    """Supported emit destinations."""

    HTTP_RESPONSE = "http"  # Default — return in HTTP response
    REDIS_STREAM = "redis_stream"  # Publish to Redis stream
    REDIS_PUBSUB = "pubsub"  # Publish to Redis pub/sub channel
    CALLBACK = "callback"  # POST to callback URL


def _parse_targets(raw: str) -> list[EmitTarget]:
    """Parse a comma-separated target string into ``EmitTarget`` values."""
    targets: list[EmitTarget] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        for member in EmitTarget:
            if member.value == token:
                targets.append(member)
                break
    return targets or [EmitTarget.HTTP_RESPONSE]


def _render_template(template: str, ctx: "YardContext") -> str:
    """Replace ``{system_id}`` / ``{node_id}`` placeholders in a template."""
    return (
        template.replace("{system_id}", ctx.system_id)
        .replace("{node_id}", ctx.node_id)
        .replace("{agent_name}", ctx.agent_name)
    )


class Emitter:
    """Emit data to all configured targets for a given invocation context."""

    def __init__(
        self,
        ctx: "YardContext",
        targets: list[EmitTarget] | None = None,
    ) -> None:
        self._ctx = ctx
        self._targets = targets or _parse_targets(
            os.environ.get("YARD_EMIT_TARGETS", "http")
        )
        self._channel_template = os.environ.get(
            "YARD_EMIT_CHANNEL", "yard:events:{system_id}"
        )
        self._stream_template = os.environ.get(
            "YARD_EMIT_STREAM", "yard:system:{system_id}:events"
        )
        self._callback_url = os.environ.get("YARD_EMIT_CALLBACK", "")
        # Accumulate events destined for the HTTP response body
        self._http_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def emit(
        self,
        data: dict,
        event_type: str = "output",
    ) -> None:
        """Emit *data* to all configured targets."""
        event = self._build_event(data, event_type)
        await self._dispatch(event)

    async def emit_progress(
        self,
        progress: float,
        detail: str = "",
    ) -> None:
        """Emit a progress update (0.0 – 1.0)."""
        payload: dict[str, Any] = {"progress": max(0.0, min(1.0, progress))}
        if detail:
            payload["detail"] = detail
        event = self._build_event(payload, "progress")
        await self._dispatch(event)

    async def emit_log(
        self,
        message: str,
        level: str = "info",
        **extra: Any,
    ) -> None:
        """Emit a structured log event."""
        payload: dict[str, Any] = {"message": message, "level": level, **extra}
        event = self._build_event(payload, "log")
        await self._dispatch(event)

    async def close(self) -> None:
        """Flush and close all targets.  Currently a no-op placeholder."""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_event(self, data: dict, event_type: str) -> dict:
        return {
            "event_type": event_type,
            "agent_name": self._ctx.agent_name,
            "invocation_id": self._ctx.invocation_id,
            "system_id": self._ctx.system_id,
            "node_id": self._ctx.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

    async def _dispatch(self, event: dict) -> None:
        """Send *event* to every configured target."""
        for target in self._targets:
            if target is EmitTarget.HTTP_RESPONSE:
                self._http_events.append(event)

            elif target is EmitTarget.REDIS_STREAM:
                await self._emit_redis_stream(event)

            elif target is EmitTarget.REDIS_PUBSUB:
                await self._emit_redis_pubsub(event)

            elif target is EmitTarget.CALLBACK:
                await self._emit_callback(event)

    async def _emit_redis_stream(self, event: dict) -> None:
        r = await self._ctx._get_redis()
        if not r:
            logger.warning("emit_redis_stream_skipped: no Redis connection")
            return
        stream_name = _render_template(self._stream_template, self._ctx)
        try:
            await r.xadd(
                stream_name,
                {
                    "event_type": event["event_type"],
                    "invocation_id": event["invocation_id"],
                    "payload": json.dumps(event["data"]),
                    "agent_name": event["agent_name"],
                    "timestamp": event["timestamp"],
                },
            )
        except Exception as exc:
            logger.warning("emit_redis_stream_failed: %s", exc)

    async def _emit_redis_pubsub(self, event: dict) -> None:
        r = await self._ctx._get_redis()
        if not r:
            logger.warning("emit_redis_pubsub_skipped: no Redis connection")
            return
        channel = _render_template(self._channel_template, self._ctx)
        try:
            await r.publish(channel, json.dumps(event))
        except Exception as exc:
            logger.warning("emit_redis_pubsub_failed: %s", exc)

    async def _emit_callback(self, event: dict) -> None:
        url = self._callback_url
        if not url:
            return
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=event)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("emit_callback_failed url=%s: %s", url, exc)
