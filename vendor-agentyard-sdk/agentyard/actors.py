"""Actor system for AgentYard.

Actors are persistent agents keyed by an id. Each actor instance:
- Has private state stored in Redis (survives across calls)
- Receives messages via a mailbox (async queue)
- Processes them one at a time (no concurrent mutation)
- Lives as long as the key exists

Usage:
    @yard.actor(name="user-session")
    async def user_session(self, message: dict) -> dict:
        # self.state is a dict-like view over Redis
        if message["type"] == "click":
            self.state["clicks"] = self.state.get("clicks", 0) + 1
        elif message["type"] == "get_clicks":
            return {"clicks": self.state.get("clicks", 0)}
        return {"ok": True}

    # Elsewhere, send a message to actor user-session/abc123:
    from agentyard import actor_ref
    ref = actor_ref("user-session", "abc123")
    result = await ref.send({"type": "click"})
"""

import asyncio
import json
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger("agentyard.actors")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@dataclass
class ActorMetadata:
    """Metadata about an actor registered via ``@yard.actor``."""

    name: str
    description: str = ""
    state_ttl: int | None = None  # None = persistent forever
    handler: Callable[..., Awaitable[Any]] | None = None


_registered_actors: dict[str, ActorMetadata] = {}


def get_registered_actors() -> dict[str, ActorMetadata]:
    """Return a copy of all actors registered in this process."""
    return dict(_registered_actors)


def actor(
    name: str,
    description: str = "",
    state_ttl: int | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to register an actor handler.

    The handler signature is ``(self, message, ctx=None) -> Awaitable[dict]``.
    ``self.state`` is an :class:`ActorState` — a dict-like proxy over Redis
    that persists between invocations for the same ``(actor_name, key)`` pair.
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        meta = ActorMetadata(
            name=name,
            description=description,
            state_ttl=state_ttl,
            handler=func,
        )
        _registered_actors[name] = meta
        func._agentyard_actor = meta  # type: ignore[attr-defined]
        return func

    return decorator


# ---------------------------------------------------------------------------
# Actor state — dict-like proxy over a Redis hash
# ---------------------------------------------------------------------------


class ActorState:
    """Dict-like facade over a Redis hash scoped to one actor instance.

    Values are JSON-encoded on write and decoded on read. Reads are served
    from an in-memory cache loaded once per invocation; writes are batched
    and flushed at the end of the handler via :meth:`flush`.
    """

    def __init__(self, redis: Any, key: str, ttl: int | None = None) -> None:
        self._redis = redis
        self._key = key
        self._ttl = ttl
        self._cache: dict[str, Any] = {}
        self._loaded = False
        self._dirty: set[str] = set()

    async def _load(self) -> None:
        if self._loaded:
            return
        if self._redis is None:
            self._loaded = True
            return
        try:
            raw = await self._redis.hgetall(self._key)
            self._cache = {k: _decode(v) for k, v in raw.items()}
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("actor_state_load_failed key=%s err=%s", self._key, e)
            self._cache = {}
        self._loaded = True

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __getitem__(self, key: str) -> Any:
        return self._cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._dirty.add(key)

    def __delitem__(self, key: str) -> None:
        self._cache.pop(key, None)
        # Mark as dirty so we HDEL on flush.
        self._dirty.add(key)

    def __iter__(self):
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        if key not in self._cache:
            self[key] = default
        return self._cache[key]

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict snapshot of the current state."""
        return dict(self._cache)

    async def flush(self) -> None:
        """Persist any pending writes to Redis."""
        if not self._dirty:
            return
        if self._redis is None:
            self._dirty.clear()
            return
        try:
            pipe = self._redis.pipeline()
            for k in list(self._dirty):
                if k in self._cache:
                    pipe.hset(self._key, k, _encode(self._cache[k]))
                else:
                    pipe.hdel(self._key, k)
            if self._ttl:
                pipe.expire(self._key, self._ttl)
            await pipe.execute()
            self._dirty.clear()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("actor_state_flush_failed key=%s err=%s", self._key, e)


def _encode(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, default=str)
    except (TypeError, ValueError):
        return str(v)


def _decode(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


# ---------------------------------------------------------------------------
# Actor instance — what gets passed to handlers as ``self``
# ---------------------------------------------------------------------------


class ActorInstance:
    """Runtime wrapper passed to actor handlers as ``self``."""

    def __init__(self, name: str, key: str, state: ActorState) -> None:
        self.name = name
        self.key = key
        self.state = state


# ---------------------------------------------------------------------------
# ActorRef — remote/local reference used to send messages
# ---------------------------------------------------------------------------


class ActorRef:
    """A reference to an actor. Used to send messages and await replies."""

    def __init__(self, name: str, key: str, redis_url: str = "") -> None:
        self.name = name
        self.key = key
        self.redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")

    async def send(self, message: dict, *, timeout: float = 30.0) -> dict:
        """Send a message to the actor and wait for a reply.

        When Redis is available the message is enqueued on the actor's
        inbox and a background runtime (on any process that registered
        the actor) will pick it up. When Redis is not configured we
        fall back to invoking the local handler directly — useful for
        tests and single-process dev loops.
        """
        if not self.redis_url:
            return await self._local_invoke(message)

        try:
            import redis.asyncio as aioredis
        except ImportError:
            return await self._local_invoke(message)

        reply_id = f"actor-reply:{_uuid_hex()}"
        envelope = {
            "reply_to": reply_id,
            "message": message,
        }
        r = aioredis.from_url(self.redis_url, decode_responses=True)
        try:
            inbox = _inbox_key(self.name, self.key)
            await r.lpush(inbox, json.dumps(envelope))
            result = await r.blpop(reply_id, timeout=int(timeout))
            if not result:
                raise ActorTimeoutError(
                    f"Actor {self.name}/{self.key} did not reply within {timeout}s"
                )
            _, payload = result
            try:
                return json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                return {"value": payload}
        finally:
            try:
                await r.aclose()
            except Exception:  # pragma: no cover - defensive
                pass

    async def _local_invoke(self, message: dict) -> dict:
        """Fallback: invoke the handler in-process (no mailbox)."""
        meta = _registered_actors.get(self.name)
        if not meta or not meta.handler:
            raise ActorNotFoundError(
                f"Actor '{self.name}' is not registered in this process"
            )
        state = ActorState(redis=None, key=_state_key(self.name, self.key), ttl=meta.state_ttl)
        await state._load()
        actor_instance = ActorInstance(name=self.name, key=self.key, state=state)
        result = await meta.handler(actor_instance, message)
        await state.flush()
        return result if isinstance(result, dict) else {"value": result}


def actor_ref(name: str, key: str) -> ActorRef:
    """Build a reference to the actor identified by ``(name, key)``."""
    return ActorRef(name=name, key=key)


# ---------------------------------------------------------------------------
# Runtime — background worker that drains inboxes and dispatches messages
# ---------------------------------------------------------------------------


class ActorRuntime:
    """Background workers that pop messages off each actor's inbox and
    dispatch them to the handler with per-key state loaded from Redis.

    One worker task is spawned per registered actor name. Each worker
    scans for inbox keys under its name and drains them. This is a
    simple polling loop — good enough for moderate throughput and keeps
    us free of Redis cluster assumptions.
    """

    def __init__(self, redis_url: str = "") -> None:
        self.redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if not self.redis_url:
            logger.debug("actor_runtime_no_redis")
            return
        if not _registered_actors:
            logger.debug("actor_runtime_no_actors")
            return
        for name in _registered_actors:
            task = asyncio.create_task(self._worker(name))
            self._tasks.append(task)
        logger.info("actor_runtime_started count=%d", len(self._tasks))

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive
                pass
        self._tasks.clear()

    async def _worker(self, name: str) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.warning("actor_runtime_redis_not_installed")
            return

        r = aioredis.from_url(self.redis_url, decode_responses=True)
        meta = _registered_actors[name]
        try:
            while not self._stop.is_set():
                pattern = _inbox_key(name, "*")
                try:
                    # Redis doesn't support BLPOP over patterns. We iterate
                    # matching inbox keys and RPOP each. A tiny sleep between
                    # scans keeps CPU flat when everything is idle.
                    drained_any = False
                    async for inbox_key in r.scan_iter(match=pattern, count=32):
                        while True:
                            envelope_raw = await r.rpop(inbox_key)
                            if not envelope_raw:
                                break
                            drained_any = True
                            try:
                                envelope = json.loads(envelope_raw)
                            except json.JSONDecodeError:
                                continue
                            key = inbox_key.split(":")[-1]
                            await self._dispatch(r, name, key, envelope, meta)
                    if not drained_any:
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # pragma: no cover - defensive
                    logger.error("actor_worker_scan_failed name=%s err=%s", name, e)
                    await asyncio.sleep(1.0)
        finally:
            try:
                await r.aclose()
            except Exception:  # pragma: no cover - defensive
                pass

    async def _dispatch(
        self,
        r: Any,
        name: str,
        key: str,
        envelope: dict,
        meta: ActorMetadata,
    ) -> None:
        state_key = _state_key(name, key)
        state = ActorState(r, state_key, ttl=meta.state_ttl)
        await state._load()
        actor_instance = ActorInstance(name=name, key=key, state=state)
        message = envelope.get("message", {}) or {}
        reply_to = envelope.get("reply_to")
        try:
            result = await meta.handler(actor_instance, message)  # type: ignore[misc]
            await state.flush()
            if reply_to:
                payload = result if result is not None else {}
                await r.lpush(reply_to, _encode(payload))
                await r.expire(reply_to, 60)
        except Exception as e:
            logger.error(
                "actor_dispatch_failed name=%s key=%s err=%s", name, key, e
            )
            if reply_to:
                try:
                    await r.lpush(reply_to, json.dumps({"error": str(e)}))
                    await r.expire(reply_to, 60)
                except Exception:  # pragma: no cover - defensive
                    pass


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ActorError(Exception):
    """Base class for actor errors."""


class ActorNotFoundError(ActorError):
    """Raised when an ActorRef references an actor not registered locally
    and there is no Redis transport available to route the message."""


class ActorTimeoutError(ActorError):
    """Raised when an actor does not reply within the send() timeout."""


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _inbox_key(name: str, key: str) -> str:
    return f"yard:actor:inbox:{name}:{key}"


def _state_key(name: str, key: str) -> str:
    return f"yard:actor:state:{name}:{key}"


def _uuid_hex() -> str:
    return secrets.token_hex(12)
