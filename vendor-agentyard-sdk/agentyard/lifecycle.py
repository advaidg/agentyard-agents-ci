"""Agent lifecycle management — startup hooks, health tracking, graceful shutdown."""

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Coroutine

logger = logging.getLogger("agentyard.lifecycle")

# Type alias for async callbacks
AsyncCallback = Callable[[], Coroutine[Any, Any, None]]

VALID_HEALTH_STATUSES = frozenset({"healthy", "degraded", "unhealthy"})


class AgentLifecycle:
    """Manages agent startup, health, and graceful shutdown.

    Usage::

        lifecycle = AgentLifecycle("my-agent")

        await lifecycle.on_startup(load_model)
        await lifecycle.on_ready(announce_ready)
        await lifecycle.on_shutdown(flush_buffers)

        # During request handling
        req_id = lifecycle.start_request()
        try:
            ...
        finally:
            lifecycle.end_request(req_id)

        # At shutdown
        await lifecycle.graceful_shutdown(timeout_seconds=30.0)
    """

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._boot_time = time.monotonic()
        self._boot_utc = time.time()

        # Health state
        self._status: str = "healthy"
        self._detail: str = ""

        # Request tracking
        self._in_flight: dict[str, float] = {}  # request_id -> start mono time
        self._total_requests: int = 0
        self._total_errors: int = 0

        # Lifecycle callbacks
        self._startup_hooks: list[AsyncCallback] = []
        self._ready_hooks: list[AsyncCallback] = []
        self._shutdown_hooks: list[AsyncCallback] = []

    # ── Startup / ready / shutdown hooks ─────────────────────────

    async def on_startup(self, callback: AsyncCallback) -> None:
        """Register and immediately run a startup hook."""
        self._startup_hooks.append(callback)
        await callback()

    async def on_ready(self, callback: AsyncCallback) -> None:
        """Register and immediately run a ready hook."""
        self._ready_hooks.append(callback)
        await callback()

    async def on_shutdown(self, callback: AsyncCallback) -> None:
        """Register a shutdown hook (runs during graceful_shutdown)."""
        self._shutdown_hooks.append(callback)

    async def graceful_shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Wait for in-flight requests to drain, then run shutdown hooks.

        If requests do not complete within *timeout_seconds*, shutdown
        proceeds anyway to avoid hanging forever.
        """
        deadline = time.monotonic() + timeout_seconds
        while self._in_flight and time.monotonic() < deadline:
            await asyncio.sleep(0.25)

        if self._in_flight:
            logger.warning(
                "shutdown_forced agent=%s in_flight=%d",
                self._agent_name,
                len(self._in_flight),
            )

        for hook in self._shutdown_hooks:
            try:
                await asyncio.wait_for(hook(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error("shutdown_hook_timeout agent=%s", self._agent_name)
            except Exception as exc:
                logger.error(
                    "shutdown_hook_error agent=%s: %s",
                    self._agent_name,
                    exc,
                    exc_info=True,
                )

    # ── Health state ─────────────────────────────────────────────

    def set_health(self, status: str, detail: str = "") -> None:
        """Set health to 'healthy', 'degraded', or 'unhealthy'."""
        if status not in VALID_HEALTH_STATUSES:
            raise ValueError(
                f"Invalid health status '{status}'. "
                f"Must be one of: {', '.join(sorted(VALID_HEALTH_STATUSES))}"
            )
        self._status = status
        self._detail = detail

    @property
    def health(self) -> dict:
        """Current health with uptime, request count, error rate."""
        uptime_s = time.monotonic() - self._boot_time
        error_rate = (
            self._total_errors / self._total_requests
            if self._total_requests > 0
            else 0.0
        )
        return {
            "status": self._status,
            "detail": self._detail,
            "agent": self._agent_name,
            "uptime_seconds": round(uptime_s, 2),
            "total_requests": self._total_requests,
            "in_flight": len(self._in_flight),
            "error_rate": round(error_rate, 4),
        }

    # ── In-flight request tracking ───────────────────────────────

    def start_request(self) -> str:
        """Mark the start of a request. Returns a unique request_id."""
        request_id = uuid.uuid4().hex
        self._in_flight[request_id] = time.monotonic()
        self._total_requests += 1
        return request_id

    def end_request(self, request_id: str, *, errored: bool = False) -> None:
        """Mark a request as complete."""
        self._in_flight.pop(request_id, None)
        if errored:
            self._total_errors += 1

    @property
    def in_flight_count(self) -> int:
        """Number of currently in-flight requests."""
        return len(self._in_flight)
