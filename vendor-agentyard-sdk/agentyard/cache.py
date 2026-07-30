"""Cache decorator for agent handlers.

Usage:
    @yard.cache(ttl=300, key=lambda input, ctx: input["url"])
    @yard.agent(name="invoice-parser")
    async def parse(input, ctx):
        ...

The cache key is derived from the user-provided callable, hashed with SHA-256
and namespaced under the agent name.
"""
import functools
import hashlib
import inspect
import json
import logging
import os
from typing import Any, Callable

logger = logging.getLogger("agentyard.cache")


def cache(
    ttl: int = 300,
    key: Callable[..., Any] | None = None,
    *,
    namespace: str | None = None,
    on_miss: str | None = None,
):
    """Decorator that caches an agent handler's output in Redis.

    Args:
        ttl: seconds to cache the result
        key: callable receiving the same args as the handler that returns
             the cache key (str or anything serializable). Defaults to the
             JSON-serialized args.
        namespace: explicit cache namespace. Defaults to the agent name.
        on_miss: optional tag value added to the result metadata on cache miss.
    """

    def decorator(func):
        agent_name = getattr(func, "_agentyard_agent_name", None) or namespace or func.__name__
        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            redis_url = os.environ.get("YARD_REDIS_URL", "")
            if not redis_url:
                return await func(*args, **kwargs)

            try:
                import redis.asyncio as aioredis
            except ImportError:
                return await func(*args, **kwargs)

            # Compute cache key
            if key:
                try:
                    cache_key_src = key(*args, **kwargs)
                except Exception as e:
                    logger.debug("cache_key_fn_failed: %s", e)
                    return await func(*args, **kwargs)
            else:
                try:
                    # Build a stable hash from the first positional arg (usually input)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    payload = {k: _safe_serialize(v) for k, v in bound.arguments.items() if k not in ("ctx", "context")}
                    cache_key_src = json.dumps(payload, sort_keys=True, default=str)
                except Exception:
                    return await func(*args, **kwargs)

            cache_key_str = str(cache_key_src) if not isinstance(cache_key_src, str) else cache_key_src
            hashed = hashlib.sha256(cache_key_str.encode()).hexdigest()[:32]
            full_key = f"yard:cache:{agent_name}:{hashed}"

            r = aioredis.from_url(redis_url, decode_responses=True)
            try:
                cached = await r.get(full_key)
                if cached is not None:
                    try:
                        parsed = json.loads(cached)
                        if isinstance(parsed, dict):
                            parsed.setdefault("_cache", {})["hit"] = True
                        return parsed
                    except json.JSONDecodeError:
                        return cached

                result = await func(*args, **kwargs)
                try:
                    serialized = json.dumps(result, default=str)
                    await r.setex(full_key, ttl, serialized)
                except (TypeError, ValueError) as e:
                    logger.debug("cache_serialize_failed: %s", e)
                return result
            finally:
                await r.aclose()

        wrapper._agentyard_cache = {"ttl": ttl, "namespace": agent_name}
        return wrapper

    return decorator


def _safe_serialize(value: Any) -> Any:
    """Convert non-serializable values to something hashable for keying."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in value.items()}
    # Pydantic model
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    # Fallback to repr
    return repr(value)
