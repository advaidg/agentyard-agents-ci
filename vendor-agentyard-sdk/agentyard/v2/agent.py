"""The @yard.agent decorator and runtime for AgentYard v2.

Agents declare WHAT they do (intent, inputs, outputs, needs, memory).
The runtime handles HOW it executes (transport, retry, validation, secrets).
"""

import asyncio
import inspect
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Type

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from agentyard.v2.config import RuntimeConfig
from agentyard.v2.context import AgentContext
from agentyard.v2.transports import (
    InputTransport,
    OutputTransport,
    AggregateInput,
    build_input_transport,
    build_output_transport,
)
from agentyard.v2.types import Resource, MemoryContract, FailurePolicy

logger = logging.getLogger("yard.agent")


class AgentRegistry:
    """Process-wide registry of declared agents."""

    def __init__(self):
        self._agents: dict[str, dict] = {}

    def register(self, name: str, meta: dict) -> None:
        self._agents[name] = meta

    def get(self, name: str) -> dict | None:
        return self._agents.get(name)

    def list(self) -> list[dict]:
        return list(self._agents.values())


_registry = AgentRegistry()


class Yard:
    """The yard.agent decorator + runtime entrypoint."""

    def agent(
        self,
        *,
        name: str,
        namespace: str,
        intent: str,
        version: str = "1.0.0",
        inputs: Type[BaseModel] | None = None,
        outputs: Type[BaseModel] | None = None,
        is_idempotent: bool = False,
        is_long_running: bool = False,
        is_pure: bool = False,
        needs: list[Resource] | None = None,
        memory: dict | MemoryContract | None = None,
        failure: FailurePolicy | None = None,
        port: int = 9000,
        image: str | None = None,
    ):
        """Declare an agent with required capability fields.

        ``image`` — the Docker image this agent runs as, published to the
        registry so K8s deploys know what to pull. Defaults to
        ``agentyard-{name}:latest``, matching the convention the deploy
        codegen already assumes when no image is registered.
        """
        def decorator(func: Callable) -> Callable:
            mem_contract = memory if isinstance(memory, dict) else (
                {"reads": memory.reads, "writes": memory.writes, "scope": memory.scope}
                if memory else {"reads": [], "writes": []}
            )
            meta = {
                "name": name,
                "namespace": namespace,
                "intent": intent,
                "version": version,
                "inputs": inputs,
                "outputs": outputs,
                "behavior": {
                    "is_idempotent": is_idempotent,
                    "is_long_running": is_long_running,
                    "is_pure": is_pure,
                },
                "needs": needs or [],
                "memory": mem_contract,
                "failure": failure or FailurePolicy(),
                "port": port,
                "image": image or f"agentyard-{name}:latest",
                "handler": func,
            }
            _registry.register(name, meta)
            func._yard_meta = meta
            return func
        return decorator

    def run(self):
        """Start the agent runtime — adaptive transports based on declared modes."""
        agents = _registry.list()
        if not agents:
            raise RuntimeError("No @yard.agent decorated functions found")
        if len(agents) > 1:
            raise RuntimeError("Multiple agents in one process not supported")

        agent_meta = agents[0]
        config = RuntimeConfig.load()

        logger.info(
            "agent_starting name=%s namespace=%s input_mode=%s output_mode=%s",
            agent_meta["name"], agent_meta["namespace"],
            config.input_mode, config.output_mode,
        )

        # Build output + input transports up-front so misconfig fails fast
        output = self._build_output(config)
        input_transport = self._build_input(config)

        app = self._build_app(agent_meta, config, output, input_transport)

        port = int(os.environ.get("YARD_PORT", os.environ.get("PORT", agent_meta["port"])))
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    # ---------- Transport builders ----------

    def _build_output(self, config: RuntimeConfig) -> OutputTransport:
        return build_output_transport(
            config.output_mode,
            redis_url=config.redis_url,
            downstream_url=config.downstream_url,
            emit_targets=config.emit_targets,
            callback_url=config.callback_url,
        )

    def _build_input(self, config: RuntimeConfig) -> InputTransport:
        return build_input_transport(
            config.input_mode,
            redis_url=config.redis_url,
            stream=config.input_stream,
            group=config.input_group,
            channels=config.subscribe_topics,
            batch_size=config.aggregate_batch_size,
            window_seconds=config.aggregate_window_seconds,
        )

    # ---------- App ----------

    def _build_app(
        self,
        meta: dict,
        config: RuntimeConfig,
        output: OutputTransport,
        input_transport: InputTransport,
    ) -> FastAPI:
        ctx = AgentContext(config, meta)
        handler = meta["handler"]
        InputModel = meta["inputs"]
        OutputModel = meta["outputs"]

        # Lazy Redis client for hybrid-mode audit mirroring. Created on first
        # use, reused for the process lifetime. Failures never break the
        # handler's sync response — audit is best-effort.
        audit_redis: dict[str, Any] = {"client": None}

        # Token-bucket rate limit for the audit stream — prevents a runaway
        # agent from flooding Redis. Defaults: 100 events/sec, burst 200.
        # Overflow events increment a "dropped" counter; ops can alert on it.
        _audit_limiter = {
            "tokens": 200.0,
            "max_tokens": 200.0,
            "refill_per_sec": 100.0,
            "last_refill": time.monotonic(),
            "dropped": 0,
        }

        def _audit_take_token() -> bool:
            now = time.monotonic()
            elapsed = now - _audit_limiter["last_refill"]
            _audit_limiter["last_refill"] = now
            _audit_limiter["tokens"] = min(
                _audit_limiter["max_tokens"],
                _audit_limiter["tokens"] + elapsed * _audit_limiter["refill_per_sec"],
            )
            if _audit_limiter["tokens"] >= 1.0:
                _audit_limiter["tokens"] -= 1.0
                return True
            _audit_limiter["dropped"] += 1
            if _audit_limiter["dropped"] % 100 == 1:
                # Log on the 1st, 101st, 201st drop to avoid log spam.
                logger.warning(
                    "audit_rate_limited dropped_total=%d",
                    _audit_limiter["dropped"],
                )
            return False

        async def _mirror_to_audit(
            result: Any,
            invocation_id: str,
            *,
            status: str = "ok",
            error: str | None = None,
            error_class: str | None = None,
        ) -> None:
            stream = config.audit_stream
            if not stream:
                return
            if not _audit_take_token():
                return
            if audit_redis["client"] is None:
                import redis.asyncio as redis_async
                audit_redis["client"] = redis_async.from_url(
                    config.redis_url, decode_responses=True
                )
            try:
                entry: dict[str, Any] = {
                    "invocation_id": invocation_id,
                    "node_id": config.node_id or meta["name"],
                    "agent": meta["name"],
                    "ts": time.time(),
                    "status": status,
                }
                if status == "ok":
                    entry["result"] = result
                else:
                    entry["error"] = error or str(result)
                    entry["error_class"] = error_class or "Exception"
                payload = json.dumps(entry, default=str)
                # Approximate MAXLEN trim — keeps audit streams bounded over
                # long-running hybrid systems without the perf hit of exact trim.
                await audit_redis["client"].xadd(
                    stream, {"data": payload}, maxlen=50000, approximate=True,
                )
            except Exception as exc:
                logger.warning("audit_mirror_failed stream=%s err=%s", stream, exc)

        failure_policy = config.failure_policy or {}
        failure_mode = failure_policy.get("mode", "retry")
        max_retries = max(0, int(failure_policy.get("max_retries", 0)))
        retry_delay_ms = max(0, int(failure_policy.get("retry_delay_ms", 500)))
        fallback_agent = failure_policy.get("fallback_agent", "")
        dlq_topic = (
            failure_policy.get("dlq_topic")
            or f"yard:sys:{config.system_id}:dlq"
        )

        async def _send_to_dlq(input_arg: Any, invocation_id: str, exc: Exception) -> None:
            """Push a failed invocation to the Redis DLQ."""
            try:
                import redis.asyncio as redis_async
                rc = redis_async.from_url(config.redis_url, decode_responses=True)
                try:
                    payload = json.dumps({
                        "invocation_id": invocation_id,
                        "agent": meta["name"],
                        "node_id": config.node_id,
                        "error_class": type(exc).__name__,
                        "error": str(exc),
                        "ts": time.time(),
                        "input": input_arg.model_dump() if isinstance(input_arg, BaseModel) else input_arg,
                    }, default=str)
                    await rc.xadd(dlq_topic, {"data": payload}, maxlen=10000, approximate=True)
                finally:
                    await rc.aclose()
                logger.warning("agent_dlq_push agent=%s invocation=%s err=%s",
                               meta["name"], invocation_id, exc)
            except Exception as dlq_exc:
                logger.error("dlq_push_failed err=%s original_err=%s", dlq_exc, exc)

        async def _call_fallback(input_arg: Any, invocation_id: str) -> Any:
            """Forward to the fallback agent — expected to be a URL or an
            in-cluster service name reachable over A2A."""
            if not fallback_agent:
                raise RuntimeError("fallback requested but no fallback_agent configured")
            target = fallback_agent if fallback_agent.startswith("http") else f"http://{fallback_agent}:9000"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{target.rstrip('/')}/invoke",
                    json={"input": input_arg.model_dump() if isinstance(input_arg, BaseModel) else input_arg},
                    headers={"X-Invocation-ID": invocation_id, "X-Fallback-Source": meta["name"]},
                )
                resp.raise_for_status()
                body = resp.json()
                return body.get("output", body)

        async def run_handler(input_arg: Any, invocation_id: str) -> Any:
            """Invoke the agent handler with retry / fallback / DLQ per the
            failure policy the compiler injected at /yard/config.yaml."""
            ctx.invocation_id = invocation_id
            sig = inspect.signature(handler)
            accepts_ctx = len(sig.parameters) > 1

            async def _invoke_once() -> Any:
                if inspect.iscoroutinefunction(handler):
                    return await handler(input_arg, ctx) if accepts_ctx else await handler(input_arg)
                return handler(input_arg, ctx) if accepts_ctx else handler(input_arg)

            attempt = 0
            last_exc: Exception | None = None
            # retry / skip / abort all share the same retry ladder; fallback
            # and compensate replace the final raise with a side-effect.
            while True:
                try:
                    result = await _invoke_once()
                    if isinstance(result, BaseModel):
                        result = result.model_dump()
                    if attempt > 0:
                        logger.info("agent_retry_succeeded attempt=%d agent=%s",
                                    attempt, meta["name"])
                    return result
                except Exception as exc:
                    last_exc = exc
                    attempt += 1
                    if failure_mode == "retry" and attempt <= max_retries:
                        # Exponential backoff with jitter — cheap, effective.
                        import random
                        delay = (retry_delay_ms / 1000) * (2 ** (attempt - 1))
                        delay *= 0.5 + random.random()
                        logger.warning(
                            "agent_retry attempt=%d/%d agent=%s err=%s delay=%.2fs",
                            attempt, max_retries, meta["name"], exc, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    break

            # Exhausted retries (or non-retry mode). Dispatch per policy.
            assert last_exc is not None
            if failure_mode in ("fallback", "compensate") and fallback_agent:
                logger.warning(
                    "agent_fallback mode=%s fallback=%s agent=%s err=%s",
                    failure_mode, fallback_agent, meta["name"], last_exc,
                )
                return await _call_fallback(input_arg, invocation_id)
            if failure_mode == "dlq" or failure_mode == "retry":
                # After exhausting retries we still persist to DLQ so ops can replay.
                await _send_to_dlq(input_arg, invocation_id, last_exc)
            if failure_mode == "skip":
                logger.warning("agent_skip agent=%s err=%s", meta["name"], last_exc)
                return {"skipped": True, "reason": str(last_exc)}
            raise last_exc

        async def _dispatch_body(payload: Any, invocation_id: str, trace_id: str) -> None:
            try:
                input_arg = self._validate_input(payload, InputModel)
                ctx._set_invocation(invocation_id, trace_id)
                result = await run_handler(input_arg, invocation_id)
                await output.deliver(result, invocation_id=invocation_id, trace_id=trace_id)
                if config.audit_stream:
                    asyncio.create_task(_mirror_to_audit(result, invocation_id))
            except Exception as exc:
                logger.exception(
                    "background_dispatch_failed agent=%s invocation=%s err=%s",
                    meta["name"], invocation_id, exc,
                )
                if config.audit_stream:
                    asyncio.create_task(_mirror_to_audit(
                        None,
                        invocation_id,
                        status="failed",
                        error=str(exc),
                        error_class=type(exc).__name__,
                    ))

        async def dispatch(payload: Any, invocation_id: str, trace_id: str = "") -> None:
            """Used by background input transports. Tracks the task so the
            graceful-shutdown path can wait for it to complete."""
            task = asyncio.create_task(_dispatch_body(payload, invocation_id, trace_id))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)
            await task

        # Track in-flight handler invocations so graceful shutdown can wait
        # for them to finish instead of dropping mid-processing work.
        in_flight: set[asyncio.Task] = set()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info(
                "agent_ready name=%s port=%s input_mode=%s output_mode=%s",
                meta["name"], meta["port"], config.input_mode, config.output_mode,
            )
            # Auto-register in the AgentYard registry on first boot so the
            # agent appears in Studio / Mission Control without the operator
            # needing to run `agentyard publish` or click Register in the UI.
            # Idempotent on restart; disable with YARD_AUTO_REGISTER=false.
            asyncio.create_task(self._register_with_registry(meta, config))
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(meta))
            await input_transport.start(dispatch)
            app.state.in_flight = in_flight  # expose for middleware / tests
            try:
                yield
            finally:
                logger.info(
                    "agent_draining name=%s in_flight=%d",
                    meta["name"], len(in_flight),
                )
                # Stop accepting new work first.
                heartbeat_task.cancel()
                await input_transport.stop()
                # Drain in-flight with a reasonable bound (K8s default grace is 30s).
                drain_timeout = float(os.environ.get("YARD_DRAIN_TIMEOUT_SECONDS", "25"))
                if in_flight:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*in_flight, return_exceptions=True),
                            timeout=drain_timeout,
                        )
                        logger.info("agent_drain_completed name=%s", meta["name"])
                    except asyncio.TimeoutError:
                        logger.warning(
                            "agent_drain_timeout name=%s outstanding=%d",
                            meta["name"], sum(1 for t in in_flight if not t.done()),
                        )
                logger.info("agent_shutting_down name=%s", meta["name"])
                await output.close()
                if audit_redis["client"] is not None:
                    try:
                        await audit_redis["client"].aclose()
                    except Exception:
                        pass

        app = FastAPI(title=meta["name"], lifespan=lifespan)

        @app.get("/")
        async def root():
            return {
                "agent": meta["name"],
                "intent": meta["intent"],
                "version": meta["version"],
                "transport_mode": config.transport_mode,
                "input_mode": config.input_mode,
                "output_mode": config.output_mode,
                "audit_enabled": bool(config.audit_stream),
            }

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "agent": meta["name"],
                "namespace": meta["namespace"],
                "transport_mode": config.transport_mode,
                "input_mode": config.input_mode,
                "output_mode": config.output_mode,
                "audit_enabled": bool(config.audit_stream),
            }

        @app.get("/.well-known/agent.json")
        async def agent_card():
            return {
                "name": meta["name"],
                "namespace": meta["namespace"],
                "version": meta["version"],
                "intent": meta["intent"],
                "behavior": meta["behavior"],
                "needs": [{"kind": r.kind.value, "name": r.name} for r in meta["needs"]],
                "memory": meta["memory"],
                "input_schema": InputModel.model_json_schema() if InputModel else None,
                "output_schema": OutputModel.model_json_schema() if OutputModel else None,
                "skills": [{"name": "default", "description": meta["intent"]}],
                "auth_schemes": ["bearer"],
                "protocol_version": "0.2",
                "transport": {
                    "mode": config.transport_mode,
                    "input": config.input_mode,
                    "output": config.output_mode,
                    "audit_enabled": bool(config.audit_stream),
                },
                "agent_card": True,
            }

        @app.post("/")
        @app.post("/invoke")
        async def invoke(request: Request):
            body = await request.json()
            input_data = body.get("input", body) if isinstance(body, dict) else body
            invocation_id = request.headers.get("X-Invocation-ID", "")
            trace_id = request.headers.get("X-Trace-ID", "")
            ctx._set_invocation(invocation_id, trace_id)

            # Aggregate mode: buffer instead of dispatching immediately
            if isinstance(input_transport, AggregateInput):
                ack = await input_transport.offer(input_data, invocation_id, trace_id)
                return {"agent": meta["name"], "aggregated": ack}

            try:
                input_arg = self._validate_input(input_data, InputModel)
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

            # Track in-flight so graceful shutdown can wait for us.
            task = asyncio.create_task(run_handler(input_arg, invocation_id))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)
            try:
                result = await task
            except Exception as exc:
                logger.exception("agent_handler_failed name=%s", meta["name"])
                # Mirror the failure to the audit stream so hybrid-mode ops
                # see unhappy-path events in the same timeline as successes.
                if config.audit_stream:
                    asyncio.create_task(_mirror_to_audit(
                        None,
                        invocation_id,
                        status="failed",
                        error=str(exc),
                        error_class=type(exc).__name__,
                    ))
                return JSONResponse(status_code=500, content={"error": str(exc)})

            delivery = await output.deliver(
                result, invocation_id=invocation_id, trace_id=trace_id,
            )
            # Hybrid mode: fire-and-forget audit mirror in parallel with the
            # sync response, so the orchestrator still drives control flow
            # but ops/compliance gets a distributed event log for free.
            if config.audit_stream:
                asyncio.create_task(_mirror_to_audit(result, invocation_id))
            # Sync mode: include result inline (back-compat envelope)
            if config.output_mode == "sync":
                return {"output": result, "agent": meta["name"]}
            return {"agent": meta["name"], "delivery": delivery}

        return app

    @staticmethod
    def _validate_input(input_data: Any, InputModel: Type[BaseModel] | None) -> Any:
        if not InputModel:
            return input_data
        try:
            if isinstance(input_data, dict):
                return InputModel(**input_data)
            return InputModel(input_data)
        except Exception as exc:
            raise ValueError(f"Input validation failed: {exc}") from exc

    async def _register_with_registry(self, meta: dict, config: RuntimeConfig) -> None:
        """Auto-register this agent in the AgentYard registry on first boot.

        Builds the registration payload from the agent's own declaration
        (the same data ``/.well-known/agent.json`` serves) and POSTs to
        ``/agents``. Idempotent: treats HTTP 409 / duplicate-name errors as
        success so restarts don't error-log.

        Can be disabled with ``YARD_AUTO_REGISTER=false`` (for agents that
        are registered out-of-band, or running in a sandbox that shouldn't
        call the registry).
        """
        if os.environ.get("YARD_AUTO_REGISTER", "true").lower() in ("false", "0", "no"):
            logger.info("agent_auto_register_disabled name=%s", meta["name"])
            return
        registry_url = os.environ.get("AGENTYARD_REGISTRY_URL", "http://registry:8001")
        # Figure out our public-facing A2A endpoint. Prefer an explicit env
        # override; otherwise fall back to ``http://{name}:{port}`` which is
        # the standard service-DNS shape in compose + K8s.
        explicit = os.environ.get("YARD_A2A_ENDPOINT")
        port = os.environ.get("YARD_PORT", os.environ.get("PORT", str(meta["port"])))
        default_endpoint = f"http://{meta['name']}:{port}"
        a2a_endpoint = explicit or default_endpoint
        # Build the agent card from the same source of truth the HTTP card endpoint uses
        InputModel = meta["inputs"]
        OutputModel = meta["outputs"]
        agent_card = {
            "name": meta["name"],
            "namespace": meta["namespace"],
            "version": meta["version"],
            "intent": meta["intent"],
            "description": meta["intent"],
            "skills": [{"name": "default", "description": meta["intent"]}],
            "auth_schemes": ["bearer"],
            "protocol_version": "0.2",
        }
        capabilities: list[str] = []
        for resource in meta.get("needs", []):
            if hasattr(resource, "kind"):
                capabilities.append(str(resource.kind.value))
        payload = {
            "name": meta["name"],
            "namespace": meta["namespace"],
            "description": meta["intent"],
            "version": meta["version"],
            "framework": "yard-v2",
            "capabilities": capabilities,
            "tags": [],
            "a2a_url": f"{a2a_endpoint}/.well-known/agent.json",
            "a2a_endpoint": a2a_endpoint,
            "agent_card": agent_card,
            "input_schema": InputModel.model_json_schema() if InputModel else None,
            "output_schema": OutputModel.model_json_schema() if OutputModel else None,
            "owner": os.environ.get("YARD_OWNER", "yard-v2"),
            "docker_image": meta.get("image"),
        }
        # Retry a few times during boot in case the registry is still coming up
        for attempt in range(5):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(f"{registry_url}/agents", json=payload)
                    if resp.status_code in (200, 201):
                        logger.info(
                            "agent_auto_registered name=%s namespace=%s endpoint=%s",
                            meta["name"], meta["namespace"], a2a_endpoint,
                        )
                        return
                    # 409 / 400 "already exists" from upsert-less registries — log once and move on
                    body = resp.text[:200]
                    if resp.status_code in (400, 409) and (
                        "exists" in body.lower() or "duplicate" in body.lower()
                    ):
                        logger.info(
                            "agent_already_registered name=%s (skipping)",
                            meta["name"],
                        )
                        return
                    logger.warning(
                        "agent_register_failed attempt=%d status=%d body=%s",
                        attempt + 1, resp.status_code, body,
                    )
            except Exception as exc:
                logger.warning(
                    "agent_register_error attempt=%d err=%s",
                    attempt + 1, exc,
                )
            await asyncio.sleep(2.0 * (attempt + 1))
        logger.warning(
            "agent_register_gave_up name=%s — add it manually in the Registry UI",
            meta["name"],
        )

    async def _heartbeat_loop(self, meta: dict):
        """Send heartbeat to registry every 30s.

        The registry's ``/agents/heartbeat`` endpoint expects ``agent_name``
        + ``port`` (matches the legacy v1 SDK shape). Sending ``name``
        previously produced a 400 and the agent never showed up as live.
        """
        registry_url = os.environ.get("AGENTYARD_REGISTRY_URL", "http://registry:8001")
        interval = int(os.environ.get("YARD_HEARTBEAT_INTERVAL", "30"))
        port = int(os.environ.get("YARD_PORT", os.environ.get("PORT", str(meta.get("port", 0)))))
        while True:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        f"{registry_url}/agents/heartbeat",
                        json={
                            "agent_name": meta["name"],
                            "port": port,
                            "status": "healthy",
                        },
                    )
            except Exception:
                pass
            await asyncio.sleep(interval)


yard = Yard()
