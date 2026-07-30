"""Typed pub/sub message bus for AgentYard agents.

Agents define topic schemas via Pydantic models, publish typed messages,
and subscribe with decorators. The transport is Redis pub/sub, but the
SDK hides it behind a clean interface.

Example:
    from pydantic import BaseModel
    from agentyard import yard, Topic

    class InvoiceProcessed(BaseModel):
        invoice_id: str
        total: float
        currency: str

    invoices_topic = Topic("invoices.processed", InvoiceProcessed)

    # Publisher
    @yard.agent(name="invoice-parser")
    async def parse(input, ctx):
        ...
        await ctx.publish(invoices_topic, InvoiceProcessed(
            invoice_id="abc", total=99.9, currency="USD"
        ))

    # Subscriber
    @yard.subscribe(invoices_topic)
    async def on_invoice_processed(msg: InvoiceProcessed, ctx):
        print(msg.invoice_id, msg.total)
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Generic, TypeVar

logger = logging.getLogger("agentyard.topics")

T = TypeVar("T")


@dataclass
class TopicMessage(Generic[T]):
    """Envelope describing a message received from a topic."""

    topic: str
    payload: T
    publisher: str
    timestamp: str  # ISO8601
    trace_id: str | None = None
    headers: dict[str, Any] | None = None


class Topic(Generic[T]):
    """A named topic with a typed payload schema.

    Topics are defined at module level and imported by both publishers and
    subscribers. The schema must be a Pydantic BaseModel.
    """

    def __init__(self, name: str, schema: type):
        self.name = name
        self.schema = schema
        # Validate that schema is a Pydantic model
        try:
            from pydantic import BaseModel
        except ImportError as exc:  # pragma: no cover - pydantic is a hard dep
            raise ImportError("Pydantic is required for Topics") from exc
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(
                f"Topic schema must be a Pydantic BaseModel, got {schema!r}"
            )

    def __repr__(self) -> str:
        return f"Topic({self.name!r})"

    def _channel(self) -> str:
        return f"yard:topic:{self.name}"


# ── Registry of subscribers for agent lifecycle integration ────────────


@dataclass
class TopicSubscription:
    """A registered handler for a specific topic."""

    topic: Topic
    handler: Callable[..., Awaitable[Any]]
    description: str = ""


_registered_subscriptions: list[TopicSubscription] = []


def get_registered_topic_subscriptions() -> list[TopicSubscription]:
    """Return a snapshot of topic subscriptions registered in this process."""
    return list(_registered_subscriptions)


def subscribe(topic: Topic, description: str = ""):
    """Decorator to register a handler for a topic.

    The handler signature is ``(message: T, ctx: YardContext) -> Awaitable``.
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        sub = TopicSubscription(topic=topic, handler=func, description=description)
        _registered_subscriptions.append(sub)
        func._agentyard_topic_subscription = sub  # type: ignore[attr-defined]
        return func

    return decorator


# ── Publisher ──────────────────────────────────────────────────────────


class TopicPublisher:
    """Publishes strongly-typed messages to Redis pub/sub channels."""

    def __init__(self, redis_url: str = "", agent_name: str = ""):
        self.redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")
        self.agent_name = agent_name

    async def publish(
        self,
        topic: Topic,
        payload: Any,
        *,
        trace_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """Publish a validated message to a topic.

        Returns ``True`` when the message was published, ``False`` when
        the transport is unavailable. Raises ``TypeError`` when the
        payload does not match the topic schema.
        """
        # Validate against schema — strict, fail fast before touching Redis.
        if not isinstance(payload, topic.schema):
            raise TypeError(
                f"Publish to {topic.name} expected {topic.schema.__name__}, "
                f"got {type(payload).__name__}"
            )
        if not self.redis_url:
            logger.debug("topic_publish_no_redis topic=%s", topic.name)
            return False

        try:
            import redis.asyncio as aioredis
        except ImportError:  # pragma: no cover
            logger.warning("redis-py not installed, topic publish skipped")
            return False

        envelope = {
            "topic": topic.name,
            "publisher": self.agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "headers": headers or {},
            "payload": payload.model_dump()
            if hasattr(payload, "model_dump")
            else dict(payload),
        }
        r = aioredis.from_url(self.redis_url, decode_responses=True)
        try:
            await r.publish(topic._channel(), json.dumps(envelope))
            return True
        except Exception as e:
            logger.error("topic_publish_failed topic=%s err=%s", topic.name, e)
            return False
        finally:
            try:
                await r.aclose()
            except Exception:  # pragma: no cover - defensive cleanup
                pass


# ── Subscriber runtime ─────────────────────────────────────────────────


class TopicSubscriber:
    """Background task that connects to Redis pub/sub and dispatches topic messages."""

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
            logger.debug("topic_subscriber_no_redis")
            return
        if not _registered_subscriptions:
            logger.debug("topic_subscriber_no_subscriptions")
            return
        self._task = asyncio.create_task(self._run())
        logger.info(
            "topic_subscriber_started agent=%s count=%d",
            self.agent_name,
            len(_registered_subscriptions),
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
        try:
            import redis.asyncio as aioredis
        except ImportError:  # pragma: no cover
            logger.warning("topic_subscriber_redis_not_installed")
            return

        r = aioredis.from_url(self.redis_url, decode_responses=True)
        pubsub = r.pubsub()
        try:
            channels = {sub.topic._channel() for sub in _registered_subscriptions}
            if not channels:
                return
            await pubsub.subscribe(*channels)

            while not self._stop.is_set():
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if not msg or msg.get("type") != "message":
                    continue
                channel = msg.get("channel", "")
                data = msg.get("data", "")
                await self._dispatch(channel, data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("topic_subscriber_error: %s", e)
        finally:
            try:
                await pubsub.aclose()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            try:
                await r.aclose()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    async def _dispatch(self, channel: str, data: Any) -> None:
        """Validate and dispatch a received message to matching subscribers."""
        if not isinstance(data, (str, bytes, bytearray)):
            return
        try:
            envelope = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            logger.warning("topic_envelope_invalid channel=%s", channel)
            return
        if not isinstance(envelope, dict):
            return
        payload_dict = envelope.get("payload", {}) or {}

        # Match every subscription whose topic channel equals this channel
        for sub in _registered_subscriptions:
            if sub.topic._channel() != channel:
                continue

            # Strict validation — invalid payloads are logged and dropped
            # rather than raised into the handler.
            try:
                typed_payload = sub.topic.schema.model_validate(payload_dict)
            except Exception as e:
                logger.warning(
                    "topic_payload_invalid topic=%s err=%s",
                    sub.topic.name,
                    e,
                )
                continue

            # Build a lightweight context for the handler. Imported lazily
            # to avoid a circular import at module load time.
            from agentyard.context import YardContext

            ctx = YardContext(
                invocation_id=envelope.get("trace_id", "") or "",
                system_id=os.environ.get("YARD_SYSTEM_ID", ""),
                node_id=os.environ.get("YARD_NODE_ID", ""),
                agent_name=self.agent_name,
                redis_url=self.redis_url,
            )
            try:
                await sub.handler(typed_payload, ctx)
            except Exception as e:
                logger.error(
                    "topic_handler_failed topic=%s err=%s",
                    sub.topic.name,
                    e,
                )
            finally:
                try:
                    await ctx.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
