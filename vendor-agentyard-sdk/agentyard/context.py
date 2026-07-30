"""Agent execution context — provides memory, tools, tracing, circuit breakers, and logging."""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import redis.asyncio as aioredis

from agentyard.circuit_breaker import CircuitBreaker
from agentyard.client_a2a import A2AClient
from agentyard.emit import Emitter
from agentyard.llm import LLMClient
from agentyard.lock import DistributedLock
from agentyard.metrics import MetricsReporter
from agentyard.prompts import PromptsClient
from agentyard.tools import ToolExecutionError, ToolNotFoundError, ToolsClient
from agentyard.tracing import Tracer
from agentyard.vector_store import VectorStoreClient


class MemoryAccessError(Exception):
    """Raised when an agent violates a memory contract.

    A memory contract is declared at the system level (see ``MemoryConfigSchema``)
    and is enforced at runtime by ``MemoryClient`` based on the agent's name and
    the operation (read/write) being attempted on a given key.
    """


class CheckpointRejectedError(Exception):
    """Raised when a human approver rejects a checkpoint.

    The exception carries the approver comment (if any) in ``.comment``
    and the approver email in ``.approved_by`` so agent code can decide
    how to react (fall back, bail out, emit logs, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        checkpoint_id: str = "",
        comment: str | None = None,
        approved_by: str | None = None,
    ) -> None:
        super().__init__(message)
        self.checkpoint_id = checkpoint_id
        self.comment = comment
        self.approved_by = approved_by


class CheckpointTimeoutError(Exception):
    """Raised when a checkpoint's approval window expires before resolution."""

    def __init__(self, message: str, *, checkpoint_id: str = "") -> None:
        super().__init__(message)
        self.checkpoint_id = checkpoint_id


class YardContext:
    """Injected into agent functions when running in a system.

    Usage:
        @yard.agent(...)
        async def my_agent(input: dict, ctx: YardContext = None) -> dict:
            # Read shared memory
            prev = await ctx.memory.get("previous_output") if ctx else None

            # Use MCP tools
            result = await ctx.tools.execute("search_code", {"q": "bug"}) if ctx else None

            # Emit progress via the emitter
            if ctx:
                await ctx.emit.emit_progress(0.5, "halfway done")

            # Emit structured log
            if ctx:
                await ctx.emit.emit_log("Processing complete", level="info")

            # Emit arbitrary data
            if ctx:
                await ctx.emit.emit({"intermediate": "data"}, event_type="step_result")

            # Distributed tracing
            with ctx.tracer.span("process_input") as span:
                span.set_attribute("input_size", len(str(input)))
                result = process(input)

            # Circuit breaker for external calls
            breaker = ctx.get_breaker("payment-api", failure_threshold=3)
            result = await breaker.call(http_client.post, url, json=payload)

            # Log (feeds into AgentYard monitoring)
            if ctx:
                ctx.log("Processing complete", level="info")

            return {"result": "..."}
    """

    def __init__(
        self,
        invocation_id: str = "",
        system_id: str = "",
        node_id: str = "",
        agent_name: str = "",
        redis_url: str = "",
        memory_strategy: str = "shared_bus",
    ):
        self.invocation_id = invocation_id
        self.system_id = system_id
        self.node_id = node_id
        self.agent_name = agent_name
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._start_time = time.monotonic()
        self.memory = MemoryClient(self)
        # ToolsClient wraps MCP tool resolution + execution. Kept on
        # ``ctx.tools`` for backward compatibility, but new code should
        # prefer ``ctx.tool(name, args)`` which adds tracer spans.
        self._tools_client = ToolsClient()
        self.tools = self._tools_client
        self._logs: list[dict] = []
        self._progress_events: list[dict] = []
        self.memory_strategy = memory_strategy
        self.emit = Emitter(self)
        self.tracer = Tracer(agent_name, redis_url=redis_url)
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self._a2a_client = A2AClient()
        # Prompt library client — resolves the registry URL from the
        # AGENTYARD_REGISTRY_URL env var that the deployment generator
        # already injects into every agent pod.
        self.prompts = PromptsClient(
            namespace=os.environ.get("YARD_NAMESPACE", "default"),
        )
        # Unified LLM client — routes to OpenAI, Anthropic, or Bedrock
        # based on model name. Lazy-imports httpx/redis so agents that
        # never touch an LLM stay lightweight.
        self.llm = LLMClient(
            redis_url=redis_url or "",
            agent_name=agent_name,
        )
        # Metrics reporter — Redis + Prometheus dual-write. Exposed on
        # ``ctx.metrics`` so agents can emit custom counters/gauges/
        # histograms and call ``record_tokens`` / ``record_cost``
        # directly when they need finer control than the report_* helpers.
        self.metrics = MetricsReporter(
            agent_name=agent_name or "unknown", redis_url=redis_url or ""
        )
        # Unified vector store client — selects a backend (memory,
        # qdrant, pgvector, chroma, pinecone) from YARD_VECTOR_BACKEND
        # so agents can call ``ctx.vector_store.query(...)`` without
        # knowing which store is wired up in this environment.
        self.vector_store = VectorStoreClient()
        # Distributed lock primitive — Redis-backed with ownership
        # tokens and auto-renewal. Exposed via ``ctx.lock(key)``.
        self._lock = DistributedLock(redis_url=redis_url or "")
        # Typed pub/sub publisher — lets agents publish Pydantic-validated
        # messages to named topics via ``ctx.publish(topic, payload)``.
        # Imported lazily to avoid a circular import with topics.py, which
        # builds its own YardContext on message dispatch.
        from agentyard.topics import TopicPublisher

        self._topic_publisher = TopicPublisher(
            redis_url=redis_url or "",
            agent_name=agent_name or "",
        )

        # Durable scheduling — ctx.schedule(...) and ctx.wait_for(...).
        from agentyard.scheduling import Scheduler

        self._scheduler = Scheduler(
            redis_url=redis_url or "",
            agent_name=agent_name or "",
        )

        # Parallel primitives — ctx.map / ctx.reduce / ctx.race / ctx.gather.
        from agentyard.parallel import ParallelPrimitives

        self._parallel = ParallelPrimitives(self)

        # ReAct reasoning loop — ctx.reason(goal=..., tools=[...]).
        from agentyard.reasoning import Reasoner

        self._reasoner = Reasoner(self)

    async def publish(
        self,
        topic: Any,
        payload: Any,
        *,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """Publish a typed message to a named topic.

        The payload must be an instance of the topic's Pydantic schema.
        Returns ``True`` when the message was delivered to the transport
        and ``False`` when Redis is not configured. Raises ``TypeError``
        on a schema mismatch so agents fail fast on contract violations.
        """
        return await self._topic_publisher.publish(
            topic,
            payload,
            trace_id=self.invocation_id or None,
            headers=headers,
        )

    def lock(
        self,
        key: str,
        *,
        timeout: float = 30.0,
        lease_seconds: int = 60,
    ):
        """Acquire a distributed lock for a critical section.

        Usage::

            async with ctx.lock("user:123", timeout=30.0):
                # Only one agent in the fleet holds this lock at a time
                ...

        The lock auto-extends its lease while the critical section runs
        and is released on exit. Falls back to a no-op when Redis is
        not configured so agents stay single-process safe by default.
        """
        return self._lock.acquire(
            key, timeout=timeout, lease_seconds=lease_seconds
        )

    # ── Durable scheduling ────────────────────────────────────────────

    async def schedule(
        self,
        topic: str,
        delay_seconds: float,
        data: Any = None,
        *,
        key: str | None = None,
    ) -> str:
        """Schedule a future event on ``yard:events:{topic}``.

        The engine's schedule worker pops it off a Redis sorted set at
        fire time and publishes the payload, where ``ctx.wait_for`` and
        ``@yard.on`` subscribers can pick it up. Returns the event id.
        """
        return await self._scheduler.schedule(
            topic, delay_seconds, data, key=key
        )

    async def cancel_scheduled(self, event_id: str) -> bool:
        """Cancel a previously scheduled event by id."""
        return await self._scheduler.cancel(event_id)

    async def wait_for(
        self,
        event_name: str,
        *,
        timeout: float = 3600.0,
    ) -> dict:
        """Suspend the agent until ``event_name`` fires.

        Returns the event payload. Raises
        :class:`agentyard.scheduling.WaitTimeoutError` on timeout.
        """
        return await self._scheduler.wait_for(event_name, timeout=timeout)

    # ── Parallel primitives ──────────────────────────────────────────

    async def map(
        self,
        agent_name: str,
        items: list,
        *,
        concurrency: int = 10,
        timeout: float = 60.0,
        fail_fast: bool = False,
    ):
        """Fan out ``ctx.call(agent_name, item)`` for every item in ``items``."""
        return await self._parallel.map(
            agent_name,
            items,
            concurrency=concurrency,
            timeout=timeout,
            fail_fast=fail_fast,
        )

    async def reduce(
        self,
        reducer_agent: str,
        items: list,
        *,
        initial: Any = None,
        chunk_size: int = 10,
    ):
        """Fold ``items`` through a reducer agent."""
        return await self._parallel.reduce(
            reducer_agent, items, initial=initial, chunk_size=chunk_size
        )

    async def race(
        self,
        agent_calls: list[tuple[str, dict]],
        *,
        timeout: float = 30.0,
    ):
        """Call multiple agents in parallel; return the first success."""
        return await self._parallel.race(agent_calls, timeout=timeout)

    async def gather(
        self,
        agent_calls: list[tuple[str, dict]],
        *,
        timeout: float = 60.0,
        return_exceptions: bool = False,
    ):
        """Call multiple agents in parallel; return all results in order."""
        return await self._parallel.gather(
            agent_calls,
            timeout=timeout,
            return_exceptions=return_exceptions,
        )

    # ── ReAct reasoning ──────────────────────────────────────────────

    async def reason(
        self,
        goal: str,
        *,
        tools: list[str] | None = None,
        tool_descriptions: dict[str, str] | None = None,
        model: str = "gpt-4o",
        max_steps: int = 6,
        temperature: float = 0.2,
    ):
        """Run a ReAct-style tool-reasoning loop until ``goal`` is met.

        Returns a :class:`agentyard.reasoning.ReasoningResult` with the
        final answer, per-step trace, and total token/cost usage.
        """
        return await self._reasoner.run(
            goal,
            tools=tools,
            tool_descriptions=tool_descriptions,
            model=model,
            max_steps=max_steps,
            temperature=temperature,
        )

    async def _get_redis(self) -> aioredis.Redis | None:
        if not self._redis and self._redis_url:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def emit_progress(self, data: dict) -> None:
        """Emit a progress event (streaming partial results).

        .. deprecated::
            Use ``ctx.emit.emit_progress(progress, detail)`` instead.
            Kept for backward compatibility.
        """
        r = await self._get_redis()
        if r:
            stream = f"yard:system:{self.system_id}:node:{self.node_id}:progress"
            await r.xadd(
                stream,
                {
                    "invocation_id": self.invocation_id,
                    "type": "agent_progress",
                    "payload": json.dumps(data),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        self._progress_events.append(data)

    def log(self, message: str, level: str = "info", **extra: Any) -> None:
        """Structured log entry — feeds into AgentYard monitoring."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "agent": self.agent_name,
            "node_id": self.node_id,
            "invocation_id": self.invocation_id,
            "msg": message,
            **extra,
        }
        self._logs.append(entry)
        print(f"[{level.upper()}] [{self.agent_name}] {message}")

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._start_time) * 1000)

    def get_breaker(self, name: str, **kwargs: Any) -> CircuitBreaker:
        """Get or create a named circuit breaker.

        Circuit breakers are cached by name so the same instance is
        reused across multiple calls within a single invocation. Pass
        keyword arguments (e.g., ``failure_threshold``, ``recovery_timeout``)
        only on first creation; they are ignored on subsequent lookups.
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers = {
                **self.circuit_breakers,
                name: CircuitBreaker(name, **kwargs),
            }
        return self.circuit_breakers[name]

    async def report_tokens(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Record LLM token usage for the current invocation.

        Emits ``agentyard_tokens_in_total`` and
        ``agentyard_tokens_out_total`` Prometheus counters labeled by
        ``agent`` and ``model``, and mirrors the same numbers to
        Redis for Mission Control.

        Usage::

            response = await client.chat.completions.create(...)
            await ctx.report_tokens(
                model="gpt-4o",
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
            )
        """
        await self.metrics.record_tokens(
            model=model, tokens_in=tokens_in, tokens_out=tokens_out
        )

    async def report_cost(self, model: str, cost_usd: float) -> None:
        """Record the USD cost of a single LLM call.

        Emits the ``agentyard_cost_usd_total`` Prometheus counter
        labeled by ``agent`` and ``model``. The LLM pricing calculation
        is up to the caller — see ``SDK_GUIDE.md`` for a reference
        implementation.
        """
        await self.metrics.record_cost(model=model, cost_usd=cost_usd)

    async def report_tool_call(
        self,
        tool: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a tool invocation outcome.

        Most agents do not need to call this directly — ``ctx.tool()``
        already emits tool metrics — but it is exposed for agents that
        wrap custom tool shims outside the SDK's MCP client.
        """
        await self.metrics.record_tool_call(
            tool=tool, status=status, duration_seconds=duration_seconds
        )

    async def tool(
        self,
        name: str,
        arguments: dict | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict:
        """Dynamically invoke an MCP tool by name.

        Resolves ``name`` through the agent's attached tool list
        (``YARD_ATTACHED_TOOLS``), then the registry, then any legacy
        sidecars. Wraps the call in a tracer span named ``tool:{name}``
        so the invocation shows up in Jaeger and Mission Control as a
        nested child under the current agent span.

        Raises
        ------
        ToolNotFoundError
            The tool is not accessible to this agent.
        ToolExecutionError
            The tool call failed at transport level or the server
            returned an error body.

        Usage::

            @yard.agent(name="researcher")
            async def research(input, ctx):
                results = await ctx.tool("web_search", {"q": input["topic"]})
                return {"sources": results.get("urls", [])}
        """
        args = arguments or {}
        started = time.monotonic()
        tool_status = "error"
        with self.tracer.span(f"tool:{name}") as span:
            span.set_attribute("tool.name", name)
            span.set_attribute("tool.arg_count", len(args))
            try:
                result = await self._tools_client.call(
                    name, args, timeout=timeout
                )
                tool_status = "success"
            except ToolNotFoundError as exc:
                tool_status = "not_found"
                span.set_attribute("tool.status", "not_found")
                span.set_attribute("tool.error", str(exc))
                raise
            except ToolExecutionError as exc:
                tool_status = "error"
                span.set_attribute("tool.status", "error")
                span.set_attribute("tool.error", str(exc))
                if exc.server_url:
                    span.set_attribute("tool.server_url", exc.server_url)
                raise
            finally:
                # Emit Prometheus tool metrics regardless of outcome so
                # the success-rate gauge on the LLM dashboard stays honest.
                duration_seconds = time.monotonic() - started
                try:
                    await self.metrics.record_tool_call(
                        tool=name,
                        status=tool_status,
                        duration_seconds=duration_seconds,
                    )
                except Exception:
                    pass
            span.set_attribute("tool.status", "success")
            return result

    async def checkpoint(
        self,
        title: str,
        description: str = "",
        payload: dict | None = None,
        approvers: list[str] | None = None,
        timeout_seconds: int = 3600,
    ) -> dict:
        """Pause execution until a human approves, rejects, or times out.

        This creates a checkpoint record in the engine and then blocks
        (polling every 2 seconds) until the checkpoint is resolved or the
        deadline is hit. Returns a dict with ``payload``, ``approval_comment``,
        ``approved_by``, and ``checkpoint_id`` on approval.

        Raises
        ------
        CheckpointRejectedError
            If a human approver rejects the checkpoint.
        CheckpointTimeoutError
            If ``timeout_seconds`` elapses without resolution.

        Usage::

            @yard.agent(name="payout-runner")
            async def run(input, ctx):
                result = await ctx.checkpoint(
                    title="Approve payout",
                    description=f"Send ${input['amount']} to {input['recipient']}",
                    payload={"amount": input["amount"], "recipient": input["recipient"]},
                    approvers=["ops@acme.com"],
                    timeout_seconds=1800,
                )
                return {"status": "sent", "note": result.get("approval_comment")}
        """
        import httpx

        engine_url = os.environ.get(
            "AGENTYARD_ENGINE_URL", "http://engine:8003"
        ).rstrip("/")

        body = {
            "invocation_id": self.invocation_id
            or "00000000-0000-0000-0000-000000000000",
            "system_id": self.system_id
            or "00000000-0000-0000-0000-000000000000",
            "node_id": self.node_id or None,
            "title": title,
            "description": description,
            "payload": payload,
            "required_approvers": approvers or ["*"],
            "timeout_seconds": int(timeout_seconds),
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{engine_url}/checkpoints", json=body)
            resp.raise_for_status()
            envelope = resp.json()
            created = envelope.get("data") if isinstance(envelope, dict) else None
            if not isinstance(created, dict) or "id" not in created:
                raise RuntimeError(
                    "Checkpoint creation returned an unexpected response"
                )
            checkpoint_id = str(created["id"])

            # Emit a log + progress event so the UI sees the pause immediately.
            try:
                await self.emit.emit_log(
                    f"Checkpoint pending: {title}",
                    level="info",
                    checkpoint_id=checkpoint_id,
                )
            except Exception:  # pragma: no cover - best effort
                pass

            deadline = time.monotonic() + float(timeout_seconds)
            poll_interval = 2.0
            last_state = created
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CheckpointTimeoutError(
                        f"Checkpoint '{title}' timed out after {timeout_seconds}s",
                        checkpoint_id=checkpoint_id,
                    )

                try:
                    poll_resp = await client.get(
                        f"{engine_url}/checkpoints/{checkpoint_id}"
                    )
                    poll_resp.raise_for_status()
                    poll_envelope = poll_resp.json()
                    state = (
                        poll_envelope.get("data")
                        if isinstance(poll_envelope, dict)
                        else None
                    )
                    if isinstance(state, dict):
                        last_state = state
                except Exception:
                    # Transient errors shouldn't kill the poll — just retry.
                    state = last_state

                status = (
                    last_state.get("status") if isinstance(last_state, dict) else None
                )
                if status == "approved":
                    return {
                        "checkpoint_id": checkpoint_id,
                        "payload": last_state.get("payload"),
                        "approval_comment": last_state.get("approval_comment"),
                        "approved_by": last_state.get("approved_by"),
                    }
                if status == "rejected":
                    raise CheckpointRejectedError(
                        f"Checkpoint '{title}' was rejected",
                        checkpoint_id=checkpoint_id,
                        comment=last_state.get("approval_comment"),
                        approved_by=last_state.get("approved_by"),
                    )
                if status == "timeout":
                    raise CheckpointTimeoutError(
                        f"Checkpoint '{title}' expired",
                        checkpoint_id=checkpoint_id,
                    )

                await asyncio.sleep(min(poll_interval, max(remaining, 0.1)))

    async def call(
        self,
        agent_name: str,
        input_data: Any,
        *,
        timeout: float = 30.0,
    ) -> Any:
        """Call another agent by name with native trace propagation.

        Resolves the target agent via the registry, routes the call
        through a per-agent circuit breaker, wraps it in a child span
        of the current trace, and propagates ``trace_parent`` and
        ``invocation_id`` headers so the downstream call appears as a
        child in the same distributed trace.

        Usage::

            @yard.agent(name="invoice-pipeline")
            async def pipeline(input, ctx):
                parsed = await ctx.call("document-extractor", {"url": input["url"]})
                scored = await ctx.call("risk-assessor", parsed)
                return scored
        """
        breaker = self.get_breaker(
            f"agent:{agent_name}", failure_threshold=5
        )

        async def _invoke() -> Any:
            with self.tracer.span(f"call:{agent_name}") as span:
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("rpc.system", "agentyard.a2a")
                started = time.monotonic()
                try:
                    result = await self._a2a_client.call(
                        agent_name,
                        input_data,
                        timeout=timeout,
                        trace_parent=span.span_id,
                        invocation_id=self.invocation_id or None,
                    )
                    duration_ms = int(
                        (time.monotonic() - started) * 1000
                    )
                    span.set_attribute("call.duration_ms", duration_ms)
                    return result
                except Exception as exc:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(exc))
                    raise

        return await breaker.call(_invoke)

    async def close(self) -> None:
        await self.emit.close()
        await self.tracer.close()
        await self.prompts.close()
        await self.metrics.close()
        if self._redis:
            await self._redis.close()


class MemoryClient:
    """Shared memory access — reads/writes based on system's memory strategy.

    When the deployment generator declares a memory schema (via the
    ``YARD_MEMORY_SCHEMA`` env var), every read/write is validated against
    the contract. ``YARD_MEMORY_OPEN`` controls whether undeclared keys
    are allowed (defaults to true so existing systems keep working).
    """

    def __init__(self, ctx: YardContext):
        self._ctx = ctx
        # Schema is loaded from env once per agent process. Deployment
        # generator serializes the system's memory schema as JSON.
        schema_json = os.environ.get("YARD_MEMORY_SCHEMA", "")
        self._schema: dict[str, dict[str, Any]] = self._parse_schema(schema_json)
        # Open mode (default true) lets agents read/write keys that aren't
        # in the schema. Strict mode (open=false) rejects them.
        self._open_mode: bool = (
            os.environ.get("YARD_MEMORY_OPEN", "true").lower() == "true"
        )

    @staticmethod
    def _parse_schema(schema_json: str) -> dict[str, dict[str, Any]]:
        if not schema_json:
            return {}
        try:
            parsed = json.loads(schema_json)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _check_access(self, key: str, operation: str) -> None:
        """Check if this agent is allowed to read/write the given key.

        ``operation`` is ``"read"`` or ``"write"``. Raises ``MemoryAccessError``
        when the agent is not in the allow list (or when the key is undeclared
        and strict mode is on).
        """
        if not self._schema:
            return  # No contract declared — fully open.
        key_schema = self._schema.get(key)
        if key_schema is None:
            if self._open_mode:
                return
            raise MemoryAccessError(
                f"Key '{key}' not declared in memory schema and open mode is disabled"
            )
        access = key_schema.get("access", {}) or {}
        allowed = access.get(operation, []) or []
        if "*" in allowed:
            return
        agent_name = self._ctx.agent_name or ""
        if agent_name and agent_name in allowed:
            return
        raise MemoryAccessError(
            f"Agent '{agent_name}' does not have {operation} access to key '{key}'. "
            f"Allowed: {allowed}"
        )

    def _schema_ttl(self, key: str) -> int | None:
        """Return the TTL declared for a key in the schema, if any."""
        if not self._schema:
            return None
        key_schema = self._schema.get(key)
        if not isinstance(key_schema, dict):
            return None
        ttl = key_schema.get("ttl")
        return int(ttl) if isinstance(ttl, (int, float)) else None

    async def get(self, key: str) -> Any:
        """Read a value from shared memory."""
        self._check_access(key, "read")
        r = await self._ctx._get_redis()
        if not r:
            return None
        # Schema may declare a TTL for this key — in which case the value lives
        # in the standalone TTL key, not the system hash. Try the TTL key first
        # so set/get round-trip cleanly when a TTL is declared.
        if self._schema_ttl(key) is not None:
            ttl_key = f"yard:system:{self._ctx.system_id}:memory:ttl:{key}"
            raw_ttl = await r.get(ttl_key)
            if raw_ttl is not None:
                try:
                    return json.loads(raw_ttl)
                except (json.JSONDecodeError, TypeError):
                    return raw_ttl
        mem_key = f"yard:system:{self._ctx.system_id}:memory"
        raw = await r.hget(mem_key, key)
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw
        return None

    async def set(self, key: str, value: Any) -> None:
        """Write a value to shared memory."""
        self._check_access(key, "write")
        if self._ctx.memory_strategy in ("isolated", "none"):
            return  # Can't write in these modes
        r = await self._ctx._get_redis()
        if not r:
            return
        # Honor schema-declared TTL by routing through the SETEX path.
        ttl = self._schema_ttl(key)
        if ttl is not None and ttl > 0:
            ttl_key = f"yard:system:{self._ctx.system_id}:memory:ttl:{key}"
            await r.setex(ttl_key, ttl, json.dumps(value))
            return
        mem_key = f"yard:system:{self._ctx.system_id}:memory"
        await r.hset(mem_key, key, json.dumps(value))

    async def get_all(self) -> dict:
        """Read entire shared memory."""
        r = await self._ctx._get_redis()
        if not r:
            return {}
        mem_key = f"yard:system:{self._ctx.system_id}:memory"
        raw = await r.hgetall(mem_key)
        result: dict[str, Any] = {}
        for k, v in raw.items():
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = v
        return result

    async def delete(self, key: str) -> None:
        """Remove a key from shared memory."""
        self._check_access(key, "write")
        r = await self._ctx._get_redis()
        if not r:
            return
        mem_key = f"yard:system:{self._ctx.system_id}:memory"
        await r.hdel(mem_key, key)
        # Also clean up any TTL-keyed sibling so contracts behave consistently.
        if self._schema_ttl(key) is not None:
            ttl_key = f"yard:system:{self._ctx.system_id}:memory:ttl:{key}"
            await r.delete(ttl_key)

    # ── TTL support ──────────────────────────────────────────────

    async def set_with_ttl(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Set a key with automatic expiration.

        Uses a standalone Redis key (SETEX) rather than a hash field,
        since hash fields do not support per-field TTL.
        """
        self._check_access(key, "write")
        if self._ctx.memory_strategy in ("isolated", "none"):
            return
        r = await self._ctx._get_redis()
        if not r:
            return
        ttl_key = f"yard:system:{self._ctx.system_id}:memory:ttl:{key}"
        await r.setex(ttl_key, ttl_seconds, json.dumps(value))

    async def get_ttl(self, key: str) -> Any:
        """Read a value set via set_with_ttl."""
        self._check_access(key, "read")
        r = await self._ctx._get_redis()
        if not r:
            return None
        ttl_key = f"yard:system:{self._ctx.system_id}:memory:ttl:{key}"
        raw = await r.get(ttl_key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    # ── Namespaced / scoped keys ─────────────────────────────────

    async def get_scoped(self, scope: str, key: str) -> Any:
        """Get from a scoped namespace (e.g., 'agent:my-agent:key')."""
        # Scoped keys are checked under their fully-qualified composite name
        # so contract authors can declare them explicitly when needed.
        self._check_access(f"{scope}:{key}", "read")
        r = await self._ctx._get_redis()
        if not r:
            return None
        scoped_hash = f"yard:system:{self._ctx.system_id}:memory:{scope}"
        raw = await r.hget(scoped_hash, key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def set_scoped(self, scope: str, key: str, value: Any) -> None:
        """Set in a scoped namespace."""
        self._check_access(f"{scope}:{key}", "write")
        if self._ctx.memory_strategy in ("isolated", "none"):
            return
        r = await self._ctx._get_redis()
        if not r:
            return
        scoped_hash = f"yard:system:{self._ctx.system_id}:memory:{scope}"
        await r.hset(scoped_hash, key, json.dumps(value))

    # ── Atomic operations ────────────────────────────────────────

    async def increment(self, key: str, amount: int = 1) -> int:
        """Atomic increment, returns new value.

        Uses a standalone Redis key for native INCRBY atomicity.
        """
        self._check_access(key, "write")
        r = await self._ctx._get_redis()
        if not r:
            return 0
        counter_key = f"yard:system:{self._ctx.system_id}:memory:counter:{key}"
        return await r.incrby(counter_key, amount)

    # ── List operations (accumulating results) ───────────────────

    async def append(self, key: str, value: Any) -> None:
        """Append to a list stored at key."""
        self._check_access(key, "write")
        if self._ctx.memory_strategy in ("isolated", "none"):
            return
        r = await self._ctx._get_redis()
        if not r:
            return
        list_key = f"yard:system:{self._ctx.system_id}:memory:list:{key}"
        await r.rpush(list_key, json.dumps(value))

    async def get_list(self, key: str) -> list:
        """Get all items in a list."""
        self._check_access(key, "read")
        r = await self._ctx._get_redis()
        if not r:
            return []
        list_key = f"yard:system:{self._ctx.system_id}:memory:list:{key}"
        raw_items = await r.lrange(list_key, 0, -1)
        result: list[Any] = []
        for raw in raw_items:
            try:
                result.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                result.append(raw)
        return result

    # ── Subscribe to changes ─────────────────────────────────────

    async def watch(
        self,
        key_pattern: str,
        callback: Callable[[str, str], Coroutine[Any, Any, None]],
    ) -> asyncio.Task:
        """Watch for changes matching pattern via Redis keyspace notifications.

        *callback* receives ``(event_type, key)`` and must be an async function.
        Returns the background task so the caller can cancel it if needed.

        Requires Redis to have keyspace notifications enabled
        (``notify-keyspace-events`` includes at least ``Kg``).
        """
        r = await self._ctx._get_redis()
        if not r:
            raise RuntimeError("Redis not available for watch")

        prefix = f"yard:system:{self._ctx.system_id}:memory"
        channel_pattern = f"__keyspace@0__:{prefix}:{key_pattern}"

        async def _listener() -> None:
            pubsub = r.pubsub()
            await pubsub.psubscribe(channel_pattern)
            try:
                async for message in pubsub.listen():
                    if message["type"] == "pmessage":
                        event_type = message["data"]
                        full_key = message["channel"].split(":", 1)[-1]
                        await callback(event_type, full_key)
            finally:
                await pubsub.punsubscribe(channel_pattern)
                await pubsub.close()

        return asyncio.create_task(_listener())
