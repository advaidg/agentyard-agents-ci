"""Auto-registration — agents call this on startup to announce themselves to AgentYard.

Usage in a FastAPI agent:

    from agentyard import yard, auto_register

    @yard.agent(name="my-agent", ...)
    def my_handler(input): ...

    app = FastAPI(lifespan=auto_register)

Or manually:

    @asynccontextmanager
    async def lifespan(app):
        await auto_register_agents()
        yield

    app = FastAPI(lifespan=lifespan)
"""

import asyncio
import os
from contextlib import asynccontextmanager

import httpx

from agentyard.decorator import get_registered_agents


def _get_registry_url() -> str:
    """Get the registry URL from env. Supports both direct and gateway URLs."""
    return os.environ.get(
        "AGENTYARD_REGISTRY_URL",
        os.environ.get("AGENTYARD_URL", "http://registry:8001"),
    )


def _make_url(base: str, path: str) -> str:
    """Build the registration URL — handles both /agents and /api/agents."""
    base = base.rstrip("/")
    # If base looks like a gateway (has /api in it or port 8080/8000), use /api/agents
    if "/api" in base:
        return f"{base}/agents"
    # Direct registry service — just /agents
    return f"{base}/agents"


async def auto_register_agents(max_retries: int = 10, retry_delay: float = 3.0) -> None:
    """Register all @yard.agent decorated agents with the AgentYard registry.

    Retries if registry isn't ready yet (common during Docker startup).
    """
    agents = get_registered_agents()
    if not agents:
        return

    registry_url = _get_registry_url()
    url = _make_url(registry_url, "/agents")

    for agent in agents:
        payload = agent.to_registration_payload()
        registered = False

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json().get("data", {})
                        agent_id = data.get("id", "?")
                        print(f"[agentyard] {agent.name} registered (id: {str(agent_id)[:8]})")
                        registered = True
                        break
                    elif resp.status_code == 409:
                        print(f"[agentyard] {agent.name} already registered")
                        registered = True
                        break
                    else:
                        print(f"[agentyard] {agent.name} registration returned {resp.status_code}")
            except httpx.ConnectError:
                print(f"[agentyard] {agent.name} — registry not ready (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"[agentyard] {agent.name} — error: {e}")

            await asyncio.sleep(retry_delay)

        if not registered:
            print(f"[agentyard] {agent.name} — could not register, running without registration")


async def auto_register_mcp_servers(max_retries: int = 10, retry_delay: float = 3.0) -> None:
    """Register all @yard.mcp_server decorated servers with AgentYard."""
    from agentyard.decorator import get_registered_mcp_servers

    servers = get_registered_mcp_servers()
    if not servers:
        return

    registry_url = _get_registry_url()

    for server in servers:
        payload = server.to_registration_payload()
        registered = False

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(f"{registry_url}/mcp/servers", json=payload)
                    if resp.status_code == 200:
                        data = resp.json().get("data", {})
                        server_id = data.get("id", "?")
                        print(f"[agentyard] MCP server '{server.name}' registered (id: {str(server_id)[:8]})")

                        # Register tools
                        for tool in server.get_tools():
                            await client.post(f"{registry_url}/mcp/tools", json={
                                "server_id": server_id,
                                "name": tool.get("name", ""),
                                "description": tool.get("description", ""),
                                "input_schema": tool.get("input_schema"),
                                "category": tool.get("category", server.category),
                            })
                        print(f"[agentyard] {len(server.get_tools())} tools registered")
                        registered = True
                        break
                    elif resp.status_code == 409:
                        print(f"[agentyard] MCP server '{server.name}' already registered")
                        registered = True
                        break
                    else:
                        print(f"[agentyard] MCP server '{server.name}' registration returned {resp.status_code}")
            except httpx.ConnectError:
                print(f"[agentyard] MCP '{server.name}' — registry not ready (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"[agentyard] MCP '{server.name}' — error: {e}")

            await asyncio.sleep(retry_delay)

        if not registered:
            print(f"[agentyard] MCP server '{server.name}' — running without registration")


@asynccontextmanager
async def auto_register(app):
    """FastAPI lifespan that auto-registers @yard.agent agents AND @yard.mcp_server servers.

    Usage:
        app = FastAPI(lifespan=auto_register)
    """
    await auto_register_agents()
    await auto_register_mcp_servers()
    yield
