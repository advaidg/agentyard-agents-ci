"""Distributed lock for AgentYard agents.

Redis-backed Redlock-ish primitive with ownership tokens and auto-renewal.

Usage:
    async with ctx.lock("user:123", timeout=30.0):
        # Critical section — only one agent holding this lock at a time
        ...
"""
import asyncio
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

logger = logging.getLogger("agentyard.lock")


class LockAcquireError(Exception):
    """Raised when we fail to acquire the lock within the timeout."""


class LockLostError(Exception):
    """Raised if the lock is lost during the critical section (expired)."""


class DistributedLock:
    def __init__(self, redis_url: str = ""):
        self.redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")

    @asynccontextmanager
    async def acquire(
        self,
        key: str,
        *,
        timeout: float = 30.0,          # how long to wait for acquisition
        lease_seconds: int = 60,         # how long the lock auto-expires
        retry_interval: float = 0.1,
    ) -> AsyncIterator[str]:
        """Acquire a distributed lock. Yields the lock token.

        The lock auto-extends as long as the critical section runs, capped by
        lease_seconds per extension. Released on exit. If held longer than
        lease_seconds without the heartbeat running, the lock expires.
        """
        if not self.redis_url:
            # No Redis — lock is a no-op (single-process safe by default)
            logger.debug("lock_no_redis_passthrough key=%s", key)
            yield "no-redis"
            return

        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise LockAcquireError("redis-py is required for ctx.lock") from e

        full_key = f"yard:lock:{key}"
        token = secrets.token_hex(16)
        r = aioredis.from_url(self.redis_url, decode_responses=True)

        heartbeat_task: asyncio.Task | None = None
        try:
            # Acquire
            deadline = time.monotonic() + timeout
            acquired = False
            while time.monotonic() < deadline:
                acquired = bool(await r.set(full_key, token, nx=True, ex=lease_seconds))
                if acquired:
                    break
                await asyncio.sleep(retry_interval)

            if not acquired:
                raise LockAcquireError(f"Failed to acquire lock '{key}' within {timeout}s")

            # Heartbeat: re-extend lease every lease_seconds/3 as long as we still own it
            async def heartbeat():
                try:
                    while True:
                        await asyncio.sleep(max(1.0, lease_seconds / 3))
                        extended = await _safe_extend(r, full_key, token, lease_seconds)
                        if not extended:
                            logger.warning("lock_heartbeat_lost key=%s", key)
                            return
                except asyncio.CancelledError:
                    raise

            heartbeat_task = asyncio.create_task(heartbeat())

            try:
                yield token
            finally:
                if heartbeat_task:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                # Atomic release — only if we still own it
                await _safe_release(r, full_key, token)
        finally:
            await r.aclose()


# Lua scripts for atomic ops
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

_EXTEND_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
else
    return 0
end
"""


async def _safe_release(r, key: str, token: str) -> bool:
    try:
        return bool(await r.eval(_RELEASE_LUA, 1, key, token))
    except Exception as e:
        logger.debug("lock_release_failed key=%s err=%s", key, e)
        return False


async def _safe_extend(r, key: str, token: str, seconds: int) -> bool:
    try:
        return bool(await r.eval(_EXTEND_LUA, 1, key, token, str(seconds)))
    except Exception as e:
        logger.debug("lock_extend_failed key=%s err=%s", key, e)
        return False
