"""AgentYard SDK Runtime — adapts agent to transport based on YARD_* env vars.

Usage:
    from agentyard import yard

    @yard.agent(name="my-agent", image="ghcr.io/me/agent:1.0")
    async def handler(input: dict) -> dict:
        return {"result": process(input)}

    yard.run()  # Starts the right adapter
"""

import asyncio
import os


def run() -> None:
    """Start the agent runtime. Auto-selects transport from YARD_TRANSPORT env var.

    Also respects YARD_OUTPUT_MODE to configure how the agent emits results:
    - sync: return response immediately (default)
    - async: return 202 Accepted, deliver result via callback
    - stream: return SSE stream of partial outputs
    """
    transport = os.environ.get("YARD_TRANSPORT", "http")
    output_mode = os.environ.get("YARD_OUTPUT_MODE", "sync")

    # Override agent metadata output_mode if env var is set
    from agentyard.decorator import get_registered_agents

    agents = get_registered_agents()
    if agents and output_mode != "sync":
        agents[0].output_mode = output_mode

    if transport == "redis-stream":
        _run_redis_stream()
    elif transport == "both":
        _run_both()
    else:
        _run_http()


def _run_http() -> None:
    """Start FastAPI HTTP server."""
    import uvicorn

    from agentyard.adapters.http_adapter import create_http_app
    from agentyard.decorator import get_registered_agents

    agents = get_registered_agents()
    if not agents:
        raise RuntimeError("No @yard.agent decorated functions found")

    app = create_http_app(agents[0])

    port = int(os.environ.get("YARD_PORT", os.environ.get("PORT", "9000")))
    uvicorn.run(app, host="0.0.0.0", port=port)


def _run_redis_stream() -> None:
    """Start Redis Stream consumer."""
    from agentyard.adapters.redis_adapter import run_redis_consumer
    from agentyard.decorator import get_registered_agents

    agents = get_registered_agents()
    if not agents:
        raise RuntimeError("No @yard.agent decorated functions found")

    asyncio.run(run_redis_consumer(agents[0]))


def _run_both() -> None:
    """Run HTTP server for health/test + Redis consumer for production traffic."""
    import threading

    t = threading.Thread(target=_run_redis_stream, daemon=True)
    t.start()

    _run_http()
