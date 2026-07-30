"""Redis Stream transport adapter — wraps agent function in a Redis consumer.

Output modes:
- sync: emit a single agent_output event (default)
- stream: emit multiple agent_chunk events followed by agent_done
- async: emit agent_complete event when finished
"""

import asyncio
import inspect
import json
import os
from datetime import datetime, timezone

import redis.asyncio as aioredis

from agentyard.context import YardContext
from agentyard.lifecycle import AgentLifecycle
from agentyard.metrics import MetricsReporter
from agentyard.middleware import run_after_hooks, run_before_hooks, run_error_hooks
from agentyard.heartbeat import start_heartbeat
from agentyard.tools import ToolsClient
from agentyard.topics import TopicSubscriber
from agentyard.validation import validate_input


async def run_redis_consumer(agent_metadata) -> None:
    """Run the agent as a Redis Stream consumer."""

    agent_func = agent_metadata._func
    agent_name = agent_metadata.name
    agent_output_mode = getattr(agent_metadata, "output_mode", "sync")

    redis_url = os.environ.get("YARD_REDIS_URL", "redis://redis:6379")
    system_id = os.environ.get("YARD_SYSTEM_ID", "")
    node_id = os.environ.get("YARD_NODE_ID", "")

    if not system_id or not node_id:
        raise RuntimeError(
            "YARD_SYSTEM_ID and YARD_NODE_ID required for redis-stream transport"
        )

    input_stream = f"yard:system:{system_id}:node:{node_id}:in"
    output_stream = f"yard:system:{system_id}:node:{node_id}:out"
    trace_stream = f"yard:system:{system_id}:trace"
    group_name = f"agent-{node_id}"
    consumer_name = f"worker-{os.getpid()}"

    r = aioredis.from_url(redis_url, decode_responses=True)
    metrics = MetricsReporter(agent_name=agent_name, redis_url=redis_url)
    lifecycle = AgentLifecycle(agent_name)

    # Check if function accepts ctx parameter
    sig = inspect.signature(agent_func)
    accepts_ctx = len(sig.parameters) > 1

    # Create consumer group (idempotent)
    try:
        await r.xgroup_create(input_stream, group_name, "$", mkstream=True)
    except aioredis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    except Exception as exc:
        print(f"[{agent_name}] WARNING: consumer group creation failed: {exc}")

    # Start heartbeat background task
    port = int(os.environ.get("YARD_PORT", os.environ.get("PORT", "9000")))
    asyncio.create_task(start_heartbeat(agent_name, port, lifecycle=lifecycle))

    # Start typed topic subscriber alongside the input stream loop so agents
    # using @yard.subscribe still receive messages on redis-stream transport.
    topic_subscriber = TopicSubscriber(agent_name=agent_name, redis_url=redis_url)
    await topic_subscriber.start()

    print(f"[{agent_name}] Redis consumer started on {input_stream}")

    while True:
        try:
            events = await r.xreadgroup(
                group_name,
                consumer_name,
                {input_stream: ">"},
                count=1,
                block=5000,
            )

            if not events:
                continue

            for _stream, messages in events:
                for msg_id, data in messages:
                    invocation_id = data.get("invocation_id", "")
                    payload = json.loads(data.get("payload", "{}"))

                    loop = asyncio.get_event_loop()
                    start = loop.time()

                    # Create context for this invocation
                    ctx = YardContext(
                        invocation_id=invocation_id,
                        system_id=system_id,
                        node_id=node_id,
                        agent_name=agent_name,
                        redis_url=redis_url,
                        memory_strategy=os.environ.get(
                            "YARD_MEMORY", "shared_bus"
                        ),
                    )
                    ctx.tools = ToolsClient()

                    try:
                        # Validate input
                        valid, error = validate_input(
                            payload, agent_metadata.input_schema
                        )
                        if not valid:
                            raise ValueError(f"Input validation failed: {error}")

                        # Before hooks
                        payload = await run_before_hooks(payload, ctx)

                        # Call agent
                        if inspect.iscoroutinefunction(agent_func):
                            if accepts_ctx:
                                result = await agent_func(payload, ctx)
                            else:
                                result = await agent_func(payload)
                        else:
                            if accepts_ctx:
                                result = agent_func(payload, ctx)
                            else:
                                result = agent_func(payload)

                        # After hooks
                        result = await run_after_hooks(payload, result, ctx)

                        duration_ms = int((loop.time() - start) * 1000)
                        await metrics.record_invocation(duration_ms, True)

                        # Determine output mode from env or agent config
                        effective_mode = os.environ.get(
                            "YARD_OUTPUT_MODE", agent_output_mode
                        )

                        if effective_mode == "stream" and hasattr(result, "__aiter__"):
                            # Streaming: emit multiple chunk events
                            chunk_idx = 0
                            async for chunk in result:
                                await r.xadd(
                                    output_stream,
                                    {
                                        "event_id": f"evt-{msg_id}-{chunk_idx}",
                                        "invocation_id": invocation_id,
                                        "source": node_id,
                                        "type": "agent_chunk",
                                        "payload": json.dumps(chunk),
                                        "chunk_index": str(chunk_idx),
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                    },
                                )
                                chunk_idx += 1
                            # Emit done event
                            await r.xadd(
                                output_stream,
                                {
                                    "event_id": f"evt-{msg_id}-done",
                                    "invocation_id": invocation_id,
                                    "source": node_id,
                                    "type": "agent_done",
                                    "total_chunks": str(chunk_idx),
                                    "duration_ms": str(duration_ms),
                                    "timestamp": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                },
                            )
                        else:
                            # Sync or async: single output event
                            event_type = (
                                "agent_complete"
                                if effective_mode == "async"
                                else "agent_output"
                            )
                            await r.xadd(
                                output_stream,
                                {
                                    "event_id": f"evt-{msg_id}",
                                    "invocation_id": invocation_id,
                                    "source": node_id,
                                    "type": event_type,
                                    "payload": json.dumps(result),
                                    "duration_ms": str(duration_ms),
                                    "timestamp": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                },
                            )

                        await r.xadd(
                            trace_stream,
                            {
                                "invocation_id": invocation_id,
                                "node_id": node_id,
                                "agent_name": agent_name,
                                "status": "completed",
                                "duration_ms": str(duration_ms),
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )

                    except Exception as e:
                        duration_ms = int((loop.time() - start) * 1000)
                        await metrics.record_invocation(duration_ms, False)
                        await run_error_hooks(payload, e, ctx)

                        await r.xadd(
                            output_stream,
                            {
                                "event_id": f"evt-{msg_id}",
                                "invocation_id": invocation_id,
                                "source": node_id,
                                "type": "agent_error",
                                "error": str(e),
                                "duration_ms": str(duration_ms),
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    finally:
                        await ctx.close()

                    await r.xack(input_stream, group_name, msg_id)

        except aioredis.ConnectionError:
            print(f"[{agent_name}] Redis connection lost, reconnecting...")
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            print(f"[{agent_name}] Consumer cancelled, shutting down")
            break
        except Exception as e:
            print(f"[{agent_name}] Consumer error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)
