"""Durable scheduling primitives.

Two complementary capabilities:

- ``ctx.schedule(topic, delay_seconds, data)`` — schedule a future event.
  Persists into a Redis sorted set keyed by fire time. A scheduler worker
  pops items as they come due and republishes them on
  ``yard:events:{topic}`` so :meth:`ctx.wait_for` and :func:`yard.on`
  subscribers receive them.

- ``ctx.wait_for(event_name, timeout)`` — suspend the current invocation
  until an event arrives. The handler resumes with the event payload.

Together these are the building blocks for long-running durable workflows
without baking a workflow engine into AgentYard. The agent code looks
straight-line; Redis carries the durable state.
"""
import asyncio
import json
import logging
import os
import secrets
import time
from typing import Any

logger = logging.getLogger("agentyard.scheduling")


class ScheduleError(Exception):
    """Raised when scheduling can't be performed (e.g. Redis missing)."""


class WaitTimeoutError(Exception):
    """Raised when ``wait_for`` exceeds its deadline."""


class Scheduler:
    """Redis-backed scheduler used by ``YardContext.schedule`` / ``wait_for``."""

    SCHEDULE_KEY = "yard:schedule"

    def __init__(self, redis_url: str = "", agent_name: str = ""):
        self.redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")
        self.agent_name = agent_name

    async def schedule(
        self,
        topic: str,
        delay_seconds: float,
        data: Any = None,
        *,
        key: str | None = None,
    ) -> str:
        """Schedule an event to fire after ``delay_seconds``.

        Returns the event id. The scheduler worker pops it off the sorted
        set at fire time and publishes the payload on
        ``yard:events:{topic}``.
        """
        if not self.redis_url:
            raise ScheduleError("ctx.schedule requires Redis (YARD_REDIS_URL)")
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ScheduleError("redis-py is required for ctx.schedule") from exc

        event_id = key or secrets.token_hex(12)
        fire_at = time.time() + max(0.0, float(delay_seconds))
        envelope = {
            "id": event_id,
            "topic": topic,
            "data": data,
            "scheduled_by": self.agent_name,
            "fire_at": fire_at,
        }
        client = aioredis.from_url(self.redis_url, decode_responses=True)
        try:
            await client.zadd(self.SCHEDULE_KEY, {json.dumps(envelope): fire_at})
            return event_id
        finally:
            await client.aclose()

    async def cancel(self, event_id: str) -> bool:
        """Cancel a previously-scheduled event by id. Returns True if removed."""
        if not self.redis_url:
            return False
        try:
            import redis.asyncio as aioredis
        except ImportError:
            return False
        client = aioredis.from_url(self.redis_url, decode_responses=True)
        try:
            members = await client.zrange(self.SCHEDULE_KEY, 0, -1)
            for raw in members:
                try:
                    env = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if env.get("id") == event_id:
                    await client.zrem(self.SCHEDULE_KEY, raw)
                    return True
            return False
        finally:
            await client.aclose()

    async def wait_for(
        self,
        event_name: str,
        *,
        timeout: float = 3600.0,
    ) -> dict:
        """Suspend until ``event_name`` fires. Returns the event payload.

        Raises :class:`WaitTimeoutError` if the deadline is reached first.
        """
        if not self.redis_url:
            raise ScheduleError("ctx.wait_for requires Redis (YARD_REDIS_URL)")
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ScheduleError("redis-py is required for ctx.wait_for") from exc

        client = aioredis.from_url(self.redis_url, decode_responses=True)
        pubsub = client.pubsub()
        channel = f"yard:events:{event_name}"
        deadline = time.time() + max(0.0, float(timeout))
        try:
            await pubsub.subscribe(channel)
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise WaitTimeoutError(
                        f"wait_for('{event_name}') exceeded {timeout}s"
                    )
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=min(5.0, remaining),
                )
                if not msg:
                    continue
                if msg.get("type") != "message":
                    continue
                payload = msg.get("data", "")
                if isinstance(payload, str):
                    try:
                        return json.loads(payload)
                    except json.JSONDecodeError:
                        return {"raw": payload}
                return {"raw": payload}
        finally:
            try:
                await pubsub.aclose()
            except Exception:
                pass
            await client.aclose()


# ── Worker loop (run inside the engine service) ─────────────────────────


async def poll_schedule_loop(redis_url: str, *, interval: float = 1.0) -> None:
    """Background loop that fires due scheduled events.

    Designed to run inside the engine service or engine-worker. On each
    tick it pops every entry whose ``fire_at`` is in the past and
    publishes the payload on ``yard:events:{topic}``.
    """
    if not redis_url:
        logger.debug("schedule_poll_no_redis")
        return
    try:
        import redis.asyncio as aioredis
    except ImportError:
        return

    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        while True:
            try:
                now = time.time()
                due = await client.zrangebyscore(
                    Scheduler.SCHEDULE_KEY, "-inf", now
                )
                for raw in due:
                    try:
                        envelope = json.loads(raw)
                    except json.JSONDecodeError:
                        await client.zrem(Scheduler.SCHEDULE_KEY, raw)
                        continue
                    topic = envelope.get("topic", "")
                    channel = f"yard:events:{topic}"
                    try:
                        await client.publish(
                            channel, json.dumps(envelope.get("data", {}))
                        )
                    finally:
                        await client.zrem(Scheduler.SCHEDULE_KEY, raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("schedule_poll_failed: %s", exc)
            await asyncio.sleep(interval)
    finally:
        await client.aclose()
