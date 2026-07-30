"""HTTP transport adapter — wraps agent function in a FastAPI server.

Output modes:
- sync: return response immediately (default)
- async: return 202 with invocation_id, result via callback URL
- stream: return SSE stream of partial outputs
"""

import asyncio
import inspect
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agentyard.context import YardContext
from agentyard.envelope import wrap_output
from agentyard.events import EventSubscriber
from agentyard.lifecycle import AgentLifecycle
from agentyard.topics import TopicSubscriber
from agentyard.metrics import MetricsReporter
from agentyard.middleware import run_after_hooks, run_before_hooks, run_error_hooks
from agentyard.heartbeat import start_heartbeat
from agentyard.register import auto_register_agents
from agentyard.tools import ToolsClient
from agentyard.validation import validate_input


def create_http_app(agent_metadata) -> FastAPI:
    """Create a FastAPI app that serves the agent function via A2A protocol."""

    agent_func = agent_metadata._func
    agent_card = agent_metadata.to_registration_payload()["agent_card"]
    agent_name = agent_metadata.name
    output_mode = getattr(agent_metadata, "output_mode", "sync")
    output_format = getattr(agent_metadata, "output_format", "json")
    max_output_size = getattr(agent_metadata, "max_output_size", 1024 * 1024)

    metrics = MetricsReporter(
        agent_name=agent_name,
        redis_url=os.environ.get("YARD_REDIS_URL", ""),
    )

    lifecycle = AgentLifecycle(agent_name)
    event_subscriber = EventSubscriber(
        agent_name=agent_name,
        redis_url=os.environ.get("YARD_REDIS_URL", ""),
    )
    topic_subscriber = TopicSubscriber(
        agent_name=agent_name,
        redis_url=os.environ.get("YARD_REDIS_URL", ""),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await auto_register_agents()
        # Start heartbeat background task
        port = int(os.environ.get("YARD_PORT", os.environ.get("PORT", "9000")))
        heartbeat_task = asyncio.create_task(
            start_heartbeat(agent_name, port, lifecycle=lifecycle)
        )
        # Start event subscriber (no-op if Redis/subscriptions missing)
        await event_subscriber.start()
        # Start typed topic subscriber (no-op if Redis/subscriptions missing)
        await topic_subscriber.start()
        yield
        heartbeat_task.cancel()
        await event_subscriber.stop()
        await topic_subscriber.stop()
        await lifecycle.graceful_shutdown()
        await metrics.close()

    app = FastAPI(title=agent_name, lifespan=lifespan)

    @app.get("/.well-known/agent.json")
    async def get_agent_card():
        return agent_card

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        """Expose Prometheus metrics for scraping.

        The agentyard SDK registers LLM-specific counters and
        histograms on the default prometheus_client registry (see
        ``agentyard.metrics``); this endpoint simply renders them.
        """
        return Response(
            content=generate_latest(), media_type=CONTENT_TYPE_LATEST
        )

    async def _invoke_agent(input_data: dict, ctx: YardContext):
        """Core agent invocation logic shared across output modes.

        Automatically wraps every invocation in a tracer span so it
        appears in Jaeger without the handler opting in.
        """
        valid, error = validate_input(input_data, agent_metadata.input_schema)
        if not valid:
            raise ValueError(f"Input validation failed: {error}")

        input_data = await run_before_hooks(input_data, ctx)

        sig = inspect.signature(agent_func)
        accepts_ctx = len(sig.parameters) > 1

        with ctx.tracer.span(f"agent.invoke:{agent_name}") as root_span:
            root_span.set_attribute("agent.name", agent_name)
            root_span.set_attribute("invocation.id", ctx.invocation_id)
            if ctx.system_id:
                root_span.set_attribute("system.id", ctx.system_id)

            if inspect.iscoroutinefunction(agent_func):
                result = await (
                    agent_func(input_data, ctx)
                    if accepts_ctx
                    else agent_func(input_data)
                )
            else:
                result = (
                    agent_func(input_data, ctx)
                    if accepts_ctx
                    else agent_func(input_data)
                )

            result = await run_after_hooks(input_data, result, ctx)

            # Enforce max output size
            serialized = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
            if len(serialized.encode()) > max_output_size:
                raise ValueError(
                    f"Output exceeds max size ({max_output_size} bytes)"
                )

            root_span.set_attribute("duration_ms", root_span.duration_ms)

        # Export the root span (and any child spans created by the handler)
        await ctx.tracer.export(root_span)

        return result

    @app.post("/")
    async def process(request: Request):
        body = await request.json()
        input_data = body.get("input", body)

        # Allow per-request output mode override via header
        req_output_mode = request.headers.get(
            "X-Output-Mode", os.environ.get("YARD_OUTPUT_MODE", output_mode)
        )
        callback_url = request.headers.get("X-Callback-URL", body.get("callback_url"))

        ctx = YardContext(
            invocation_id=request.headers.get("X-Invocation-ID", ""),
            system_id=os.environ.get("YARD_SYSTEM_ID", ""),
            node_id=os.environ.get("YARD_NODE_ID", ""),
            agent_name=agent_name,
            redis_url=os.environ.get("YARD_REDIS_URL", ""),
            memory_strategy=os.environ.get("YARD_MEMORY", "shared_bus"),
        )
        ctx.tools = ToolsClient()

        start = time.monotonic()

        if req_output_mode == "stream":
            # SSE stream of partial outputs
            async def stream_generator():
                try:
                    result = await _invoke_agent(input_data, ctx)
                    # If result is iterable (generator/list), stream items
                    if hasattr(result, "__aiter__"):
                        async for chunk in result:
                            yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"
                    elif isinstance(result, list):
                        for item in result:
                            yield f"data: {json.dumps({'type': 'chunk', 'data': item})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'chunk', 'data': result})}\n\n"

                    duration_ms = int((time.monotonic() - start) * 1000)
                    await metrics.record_invocation(duration_ms, True)
                    yield f"data: {json.dumps({'type': 'done', 'duration_ms': duration_ms})}\n\n"
                except Exception as e:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    await metrics.record_invocation(duration_ms, False)
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                finally:
                    await ctx.close()

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        if req_output_mode == "async":
            # Return 202 immediately, process in background, POST result to callback
            invocation_id = str(uuid.uuid4())

            async def _background_run():
                try:
                    result = await _invoke_agent(input_data, ctx)
                    duration_ms = int((time.monotonic() - start) * 1000)
                    await metrics.record_invocation(duration_ms, True)

                    if callback_url:
                        import httpx

                        async with httpx.AsyncClient(timeout=10.0) as client:
                            await client.post(callback_url, json={
                                "invocation_id": invocation_id,
                                "output": result,
                                "status": "completed",
                                "duration_ms": duration_ms,
                            })
                except Exception as e:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    await metrics.record_invocation(duration_ms, False)
                    if callback_url:
                        import httpx

                        async with httpx.AsyncClient(timeout=10.0) as client:
                            await client.post(callback_url, json={
                                "invocation_id": invocation_id,
                                "error": str(e),
                                "status": "failed",
                                "duration_ms": duration_ms,
                            })
                finally:
                    await ctx.close()

            asyncio.create_task(_background_run())
            return JSONResponse(
                status_code=202,
                content={"invocation_id": invocation_id, "status": "accepted"},
            )

        # Default: sync mode
        req_id = lifecycle.start_request()
        try:
            result = await _invoke_agent(input_data, ctx)

            # G2: auto-promote to streaming if the handler returned an
            # async generator. This lets agents simply `yield chunk` and
            # callers see SSE without setting output_mode.
            if inspect.isasyncgen(result) or hasattr(result, "__aiter__"):

                async def _auto_stream():
                    final_chunks: list = []
                    try:
                        async for chunk in result:  # type: ignore[union-attr]
                            final_chunks.append(chunk)
                            payload = chunk if isinstance(chunk, dict) else {"delta": chunk}
                            yield f"data: {json.dumps(payload, default=str)}\n\n"
                        duration_ms = int((time.monotonic() - start) * 1000)
                        await metrics.record_invocation(duration_ms, True)
                        lifecycle.end_request(req_id)
                        final_payload = {
                            "done": True,
                            "duration_ms": duration_ms,
                            "final": final_chunks if len(final_chunks) > 1 else (final_chunks[0] if final_chunks else None),
                        }
                        yield f"data: {json.dumps(final_payload, default=str)}\n\n"
                    except Exception as e:
                        duration_ms = int((time.monotonic() - start) * 1000)
                        await metrics.record_invocation(duration_ms, False)
                        lifecycle.end_request(req_id, errored=True)
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    finally:
                        await ctx.close()

                return StreamingResponse(
                    _auto_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            duration_ms = int((time.monotonic() - start) * 1000)
            await metrics.record_invocation(duration_ms, True)
            lifecycle.end_request(req_id)

            envelope = wrap_output(
                result,
                agent_name=agent_name,
                invocation_id=ctx.invocation_id,
                system_id=ctx.system_id,
                node_id=ctx.node_id,
                duration_ms=duration_ms,
            )
            return envelope.to_dict()

        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            await metrics.record_invocation(duration_ms, False)
            lifecycle.end_request(req_id, errored=True)
            await run_error_hooks(input_data, e, ctx)
            return JSONResponse(
                status_code=500, content={"error": str(e)}
            )
        finally:
            await ctx.close()

    @app.get("/health")
    async def health():
        health_data = lifecycle.health
        health_data["transport"] = "http"
        return health_data

    return app
