"""Agent heartbeat -- periodic health reporting to AgentYard registry."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from agentyard.lifecycle import AgentLifecycle

logger = logging.getLogger("agentyard.heartbeat")

DEFAULT_INTERVAL_SECONDS = 30
MAX_BACKOFF_SECONDS = 60


async def start_heartbeat(
    agent_name: str,
    port: int,
    interval: int | None = None,
    lifecycle: AgentLifecycle | None = None,
) -> None:
    """Send periodic heartbeats to the registry.

    The heartbeat updates last_seen in Redis so the platform knows the agent
    is alive without polling the A2A health endpoint.

    Args:
        agent_name: Registered agent name.
        port: Port the agent is listening on.
        interval: Seconds between heartbeats.  Falls back to
            ``YARD_HEARTBEAT_INTERVAL`` env var, then to 30s.
        lifecycle: Optional ``AgentLifecycle`` instance.  When provided the
            heartbeat payload includes the real health status instead of a
            static ``"alive"`` string.
    """
    registry_url = os.environ.get("AGENTYARD_REGISTRY_URL", "")
    if not registry_url:
        return

    resolved_interval = interval or int(
        os.environ.get("YARD_HEARTBEAT_INTERVAL", str(DEFAULT_INTERVAL_SECONDS))
    )

    consecutive_failures = 0

    while True:
        status = "alive"
        detail = ""
        if lifecycle is not None:
            health = lifecycle.health
            status = health.get("status", "alive")
            detail = health.get("detail", "")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{registry_url}/agents/heartbeat",
                    json={
                        "agent_name": agent_name,
                        "port": port,
                        "status": status,
                        "detail": detail,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                response.raise_for_status()

            if consecutive_failures > 0:
                logger.info(
                    "Heartbeat recovered after %d consecutive failure(s)",
                    consecutive_failures,
                )
            consecutive_failures = 0
            logger.debug("Heartbeat sent for %s (status=%s)", agent_name, status)

        except Exception as exc:
            consecutive_failures += 1
            logger.warning(
                "Heartbeat failed for %s (attempt #%d): %s",
                agent_name,
                consecutive_failures,
                exc,
            )

        # Exponential backoff on consecutive failures, capped at MAX_BACKOFF_SECONDS
        if consecutive_failures > 0:
            backoff = min(
                resolved_interval * (2 ** (consecutive_failures - 1)),
                MAX_BACKOFF_SECONDS,
            )
            await asyncio.sleep(backoff)
        else:
            await asyncio.sleep(resolved_interval)
