"""Event subscription system for agents.

Agents can subscribe to events from other agents, memory changes, or system events.

Usage:
    from agentyard import yard

    @yard.on("agent:invoice-processor:complete")
    async def on_invoice_done(event, ctx):
        await ctx.emit.log(f"Invoice {event['invoice_id']} ready")

    @yard.on("memory:changed:user_profile:*")
    async def on_profile_change(event, ctx):
        ...

    @yard.on("system:deployed")
    async def on_deploy(event, ctx):
        ...

Event transport is Redis pub/sub (channels matching the event pattern).
Subscriptions are auto-registered on agent startup and listed in the agent card.
"""

import asyncio
import fnmatch
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger("agentyard.events")


@dataclass
class EventSubscription:
    """A registered event handler subscription."""

    pattern: str  # Redis channel pattern (supports glob wildcards)
    handler: Callable[..., Awaitable[Any]]
    description: str = ""


# Global registry of subscriptions on this agent process
_subscriptions: list[EventSubscription] = []


def get_registered_subscriptions() -> list[EventSubscription]:
    return list(_subscriptions)


def on(pattern: str, description: str = ""):
    """Decorator that subscribes an async function to events matching `pattern`.

    Pattern examples:
        - "agent:invoice-processor:complete" — exact match
        - "agent:*:complete"                 — any agent completion
        - "memory:changed:user_profile:*"    — memory key changes under user_profile
        - "system:deployed"                  — platform-level events
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        sub = EventSubscription(
            pattern=pattern,
            handler=func,
            description=description or (func.__doc__ or "").strip().split("\n")[0],
        )
        _subscriptions.append(sub)
        func._agentyard_subscription = sub  # type: ignore[attr-defined]
        return func

    return decorator


# ---------------------------------------------------------------------------
# Runtime subscriber — called from adapter lifespan
# ---------------------------------------------------------------------------


class EventSubscriber:
    """Background task that connects to Redis pub/sub and dispatches events.

    Redis keyspace notifications must be enabled for memory:* events:
        CONFIG SET notify-keyspace-events KEA
    """

    def __init__(self, agent_name: str, redis_url: str = ""):
        self.agent_name = agent_name
        self.redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Start the subscriber background task.

        Silently skips if Redis is not configured or no subscriptions exist.
        """
        if not self.redis_url:
            logger.debug("event_subscriber_no_redis")
            return
        if not _subscriptions:
            logger.debug("event_subscriber_no_subscriptions")
            return

        self._task = asyncio.create_task(self._run())
        logger.info(
            "event_subscriber_started agent=%s count=%d",
            self.agent_name,
            len(_subscriptions),
        )

    async def stop(self) -> None:
        """Cancel the subscriber task cleanly."""
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        """Main subscription loop.

        Uses Redis PSUBSCRIBE to listen for events matching any registered pattern.
        Dispatches matching events to handlers.
        """
        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.warning("event_subscriber_redis_not_installed")
            return

        redis = aioredis.from_url(self.redis_url, decode_responses=True)

        try:
            pubsub = redis.pubsub()

            # Build distinct Redis patterns from our subscription list.
            # Redis glob syntax is simpler than fnmatch but overlaps enough for our cases.
            patterns = {_to_redis_pattern(sub.pattern) for sub in _subscriptions}
            if patterns:
                await pubsub.psubscribe(*patterns)

            # Also enable memory:changed:* via keyspace notifications if subscribed.
            has_memory_sub = any(
                sub.pattern.startswith("memory:") for sub in _subscriptions
            )
            if has_memory_sub:
                try:
                    await redis.config_set("notify-keyspace-events", "KEA")
                    await pubsub.psubscribe("__keyspace@0__:*")
                except Exception:
                    # Config set may be disallowed in managed Redis
                    pass

            while not self._stop.is_set():
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if not msg or msg["type"] not in ("pmessage", "message"):
                    continue

                channel = msg.get("channel", "")
                data = msg.get("data", "")
                await self._dispatch(channel, data)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("event_subscriber_error: %s", e)
        finally:
            try:
                await pubsub.aclose()
            except Exception:
                pass
            await redis.aclose()

    async def _dispatch(self, channel: str, data: Any) -> None:
        """Call all subscriptions whose pattern matches the received channel."""
        # Parse data as JSON if possible
        event: dict[str, Any]
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                event = parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                event = {"value": data}
        elif isinstance(data, dict):
            event = data
        else:
            event = {"value": data}

        # Add channel metadata
        event["_channel"] = channel

        # Translate keyspace events into memory:changed:{key} form
        effective_channel = channel
        if channel.startswith("__keyspace@0__:"):
            key = channel.split(":", 1)[1]
            effective_channel = f"memory:changed:{key}"
            event["_key"] = key
            event["_op"] = event.get("value", "")

        # Build a lightweight context
        from agentyard.context import YardContext

        ctx = YardContext(
            invocation_id="",
            system_id=os.environ.get("YARD_SYSTEM_ID", ""),
            node_id=os.environ.get("YARD_NODE_ID", ""),
            agent_name=self.agent_name,
            redis_url=self.redis_url,
        )

        try:
            for sub in _subscriptions:
                if fnmatch.fnmatchcase(effective_channel, sub.pattern):
                    try:
                        await sub.handler(event, ctx)
                    except Exception as e:
                        logger.error(
                            "event_handler_failed pattern=%s channel=%s error=%s",
                            sub.pattern,
                            effective_channel,
                            e,
                        )
        finally:
            await ctx.close()


def _to_redis_pattern(pattern: str) -> str:
    """Convert our glob pattern to a Redis PSUBSCRIBE pattern.

    Both support * as a wildcard, but our dispatch uses fnmatch for precise
    matching so the Redis pattern just needs to be a superset.
    """
    return pattern
