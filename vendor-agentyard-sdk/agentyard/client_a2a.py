"""Client for calling other A2A agents from within an agent handler.

This powers ``ctx.call(agent_name, input)`` — native agent-to-agent
calls that resolve targets by name via the AgentYard registry and
forward requests over HTTP using the global bearer token.

Trace context is propagated via ``X-Trace-Parent`` and invocation
identity via ``X-Invocation-ID`` so multi-agent calls stitch into a
single distributed trace.
"""

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("agentyard.a2a_client")

DEFAULT_REGISTRY_URL = "http://registry:8001"
DEFAULT_TIMEOUT_SECONDS = 30.0
RESOLVE_TIMEOUT_SECONDS = 5.0
RESOLVE_MAX_RETRIES = 3
MAX_CALL_DEPTH = 10

_call_depth: int = 0


class A2ACallError(Exception):
    """Raised when an agent-to-agent call cannot be completed."""


class A2AClient:
    """HTTP client for invoking other A2A agents by name.

    Resolves agents against the registry (results cached in-process)
    and POSTs to their ``a2a_endpoint`` with the shared bearer token.
    Uses a shared connection pool to avoid TCP exhaustion under load.
    """

    def __init__(self, registry_url: str = "", token: str = "") -> None:
        self.registry_url = (
            registry_url
            or os.environ.get("AGENTYARD_REGISTRY_URL", DEFAULT_REGISTRY_URL)
        )
        self.token = token or os.environ.get("YARD_GLOBAL_TOKEN", "")
        self._agent_cache: dict[str, dict[str, Any]] = {}
        self._pool: httpx.AsyncClient | None = None

    def _get_pool(self) -> httpx.AsyncClient:
        if self._pool is None or self._pool.is_closed:
            self._pool = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10, keepalive_expiry=30.0),
                timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS, connect=5.0),
            )
        return self._pool

    async def resolve(self, agent_name: str) -> dict[str, Any] | None:
        """Look up an agent by name in the registry (cached in-process).

        Retries on transient errors (timeout, connection) with exponential
        backoff. Returns the full agent record dict, or ``None`` if no agent
        with an exact name match is found.
        """
        if agent_name in self._agent_cache:
            return self._agent_cache[agent_name]

        last_exc: Exception | None = None
        for attempt in range(RESOLVE_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=RESOLVE_TIMEOUT_SECONDS, limits=httpx.Limits(max_connections=10)) as client:
                    resp = await client.get(
                        f"{self.registry_url}/agents",
                        params={"q": agent_name, "limit": 10},
                    )
                    if resp.status_code != 200:
                        logger.warning(
                            "registry lookup failed for %s: HTTP %s",
                            agent_name,
                            resp.status_code,
                        )
                        return None
                    body = resp.json()
                    data = body.get("data") or {}
                    items = data.get("items") or []
                    for item in items:
                        if item.get("name") == agent_name:
                            self._agent_cache = {
                                **self._agent_cache,
                                agent_name: item,
                            }
                            return item
                    return None
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt < RESOLVE_MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as exc:
                logger.warning("registry lookup error for %s: %s", agent_name, exc)
                return None

        logger.warning(
            "registry lookup exhausted retries for %s: %s", agent_name, last_exc
        )
        return None

    async def call(
        self,
        agent_name: str,
        input_data: Any,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        trace_parent: str | None = None,
        invocation_id: str | None = None,
    ) -> Any:
        """Call an agent by name and return its unwrapped output.

        Raises :class:`A2ACallError` on resolution failure, HTTP
        errors, missing A2A endpoint, or circular call depth exceeded.
        """
        global _call_depth
        _call_depth += 1
        if _call_depth > MAX_CALL_DEPTH:
            _call_depth -= 1
            raise A2ACallError(
                f"Call depth exceeded ({MAX_CALL_DEPTH}): possible circular "
                f"dependency calling '{agent_name}'"
            )
        try:
            return await self._do_call(
                agent_name, input_data,
                timeout=timeout, trace_parent=trace_parent,
                invocation_id=invocation_id,
            )
        finally:
            _call_depth -= 1

    async def _do_call(
        self,
        agent_name: str,
        input_data: Any,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        trace_parent: str | None = None,
        invocation_id: str | None = None,
    ) -> Any:
        agent = await self.resolve(agent_name)
        if not agent:
            raise A2ACallError(
                f"Agent '{agent_name}' not found in registry"
            )

        endpoint = agent.get("a2a_endpoint", "")
        if not endpoint:
            raise A2ACallError(
                f"Agent '{agent_name}' has no a2a_endpoint"
            )

        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if trace_parent:
            headers["X-Trace-Parent"] = trace_parent
        if invocation_id:
            headers["X-Invocation-ID"] = invocation_id

        try:
            pool = self._get_pool()
            resp = await pool.post(
                endpoint,
                json={"input": input_data},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPStatusError as exc:
            raise A2ACallError(
                f"Agent '{agent_name}' returned HTTP "
                f"{exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise A2ACallError(
                f"HTTP error calling agent '{agent_name}': {exc}"
            ) from exc

        if isinstance(body, dict) and "output" in body:
            return body["output"]
        return body
