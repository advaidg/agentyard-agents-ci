"""MCP Tools client — calls sidecar MCP servers from within an agent.

Supports three discovery modes:
  1. Attached tools: YARD_ATTACHED_TOOLS env var — JSON list of
     ``{tool_name, server_id, server_url, config}`` injected by the
     deployment generator for tools bound to this agent node.
  2. API key mode: YARD_MCP_TOKEN + AGENTYARD_REGISTRY_URL — discovers
     tools via registry token exchange.
  3. Sidecar mode: YARD_MCP_TOOLS env var (e.g. ``github:3100,slack:3101``).

Agents should prefer the high-level ``ctx.tool(name, args)`` entrypoint on
``YardContext`` — it wraps ``ToolsClient.call`` with a tracer span.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class ToolNotFoundError(Exception):
    """Raised when a tool name cannot be resolved to a concrete MCP server.

    The tool is not in the agent's ``YARD_ATTACHED_TOOLS`` list, not
    discoverable via the registry token, and not available on any
    configured sidecar. Catch this to implement graceful fallback.
    """


class ToolExecutionError(Exception):
    """Raised when a resolved MCP tool call fails.

    Wraps transport errors (network, HTTP status), tool-reported errors
    in the response body, and timeouts. The ``tool_name`` and
    ``server_url`` attributes are populated when available.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        server_url: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.server_url = server_url
        self.__cause__ = cause


@dataclass(frozen=True)
class _ResolvedTool:
    """A tool that has been mapped to a concrete MCP server endpoint."""

    tool_name: str
    server_url: str
    server_id: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # "attached" | "discovered" | "sidecar"


# Cache lifetime for tools resolved from the registry. Attached-env tools
# are cached for the lifetime of the process because they are injected
# at deployment time and cannot change.
_REGISTRY_CACHE_TTL_SECONDS = 300


class ToolsClient:
    """Client for calling MCP tools.

    Resolution order for ``call(tool_name, args)``:
      1. ``YARD_ATTACHED_TOOLS`` — deployment-injected list of tools
         bound to this specific agent node.
      2. Registry lookup via ``YARD_MCP_TOKEN`` (cached for 5 minutes).
      3. Legacy sidecar discovery via ``YARD_MCP_TOOLS``.

    Attached tools take precedence and are always available offline.
    """

    def __init__(self) -> None:
        # Sidecar mode: legacy env var format "github:3100,slack:3101".
        tools_env = os.environ.get("YARD_MCP_TOOLS", "")
        self._servers: dict[str, int] = {}
        if tools_env:
            for entry in tools_env.split(","):
                parts = entry.strip().split(":")
                if len(parts) == 2:
                    try:
                        self._servers[parts[0]] = int(parts[1])
                    except ValueError:
                        continue

        # Registry token mode.
        self._token = os.environ.get("YARD_MCP_TOKEN", "")
        self._registry_url = os.environ.get(
            "AGENTYARD_REGISTRY_URL",
            os.environ.get("AGENTYARD_URL", ""),
        )
        self._agent_id = os.environ.get("YARD_AGENT_ID", "")

        # Parse the deployment-injected attached tools map.
        self._attached: dict[str, _ResolvedTool] = self._parse_attached_env()

        # Cache of tools resolved via registry discovery/lookup.
        self._registry_cache: dict[str, _ResolvedTool] = {}
        self._registry_cache_loaded_at: float = 0.0

        # Legacy field preserved so the old ``execute`` path keeps working.
        self._discovered_tools: list[dict] | None = None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_attached_env() -> dict[str, "_ResolvedTool"]:
        """Parse YARD_ATTACHED_TOOLS into a tool_name→_ResolvedTool map."""
        raw = os.environ.get("YARD_ATTACHED_TOOLS", "").strip()
        if not raw:
            return {}
        try:
            entries = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(entries, list):
            return {}
        resolved: dict[str, _ResolvedTool] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("tool_name") or entry.get("name")
            server_url = entry.get("server_url")
            if not name or not server_url:
                continue
            resolved[name] = _ResolvedTool(
                tool_name=name,
                server_url=str(server_url),
                server_id=str(entry.get("server_id", "")),
                config=dict(entry.get("config") or {}),
                source="attached",
            )
        return resolved

    def _registry_cache_fresh(self) -> bool:
        if not self._registry_cache:
            return False
        age = time.monotonic() - self._registry_cache_loaded_at
        return age < _REGISTRY_CACHE_TTL_SECONDS

    async def _resolve_from_registry(
        self, tool_name: str
    ) -> _ResolvedTool | None:
        """Look up a tool via the registry — first by token discovery,
        then by direct ``/mcp/tools`` search.

        Populates the process-local cache so repeated lookups inside a
        single invocation don't hammer the registry.
        """
        if self._registry_cache_fresh() and tool_name in self._registry_cache:
            return self._registry_cache[tool_name]

        if not self._registry_url:
            return None

        base = self._registry_url.rstrip("/")

        # Token-based discovery returns only tools this agent has access to.
        if self._token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{base}/mcp/tools/discover",
                        headers={"X-MCP-Token": self._token},
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", []) or []
                        self._refresh_registry_cache(data)
                        self._discovered_tools = data
                        if tool_name in self._registry_cache:
                            return self._registry_cache[tool_name]
            except httpx.HTTPError:
                pass

        # Fall back to an unauthenticated tool search by name. This path
        # is best-effort and exists so local dev without tokens still
        # resolves tools. It returns a match only when the tool is in
        # the registry and the server row exposes a ``url``.
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{base}/mcp/tools", params={"q": tool_name}
                )
                if resp.status_code != 200:
                    return None
                items = resp.json().get("data", []) or []
                for item in items:
                    if item.get("name") != tool_name:
                        continue
                    server_url = item.get("server_url") or item.get(
                        "url"
                    ) or ""
                    if not server_url and item.get("server_id"):
                        server_url = await self._fetch_server_url(
                            client, base, str(item["server_id"])
                        )
                    if not server_url:
                        continue
                    resolved = _ResolvedTool(
                        tool_name=tool_name,
                        server_url=str(server_url),
                        server_id=str(item.get("server_id", "")),
                        source="discovered",
                    )
                    self._registry_cache = {
                        **self._registry_cache,
                        tool_name: resolved,
                    }
                    self._registry_cache_loaded_at = time.monotonic()
                    return resolved
        except httpx.HTTPError:
            return None
        return None

    @staticmethod
    async def _fetch_server_url(
        client: httpx.AsyncClient, base: str, server_id: str
    ) -> str:
        try:
            resp = await client.get(f"{base}/mcp/servers/{server_id}")
            if resp.status_code == 200:
                return str(resp.json().get("data", {}).get("url", "") or "")
        except httpx.HTTPError:
            pass
        return ""

    def _refresh_registry_cache(self, entries: list[dict]) -> None:
        """Rebuild the registry cache from a discovery response."""
        new_cache: dict[str, _ResolvedTool] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            server_url = entry.get("server_url") or entry.get("url")
            if not name or not server_url:
                continue
            new_cache[name] = _ResolvedTool(
                tool_name=str(name),
                server_url=str(server_url),
                server_id=str(entry.get("server_id", "")),
                source="discovered",
            )
        if new_cache:
            self._registry_cache = new_cache
            self._registry_cache_loaded_at = time.monotonic()

    async def _resolve(self, tool_name: str) -> _ResolvedTool | None:
        """Resolve a tool name through env → registry → sidecar."""
        if tool_name in self._attached:
            return self._attached[tool_name]

        resolved = await self._resolve_from_registry(tool_name)
        if resolved is not None:
            return resolved

        # Legacy sidecar fallback: check each locally-configured server
        # for the tool by listing its tools. We return a placeholder
        # _ResolvedTool pointing at localhost so ``call`` can reuse
        # the same HTTP path.
        for server_name, port in self._servers.items():
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"http://localhost:{port}/mcp")
                    if resp.status_code != 200:
                        continue
                    tools = resp.json().get("tools", []) or []
                    if any(t.get("name") == tool_name for t in tools):
                        return _ResolvedTool(
                            tool_name=tool_name,
                            server_url=f"http://localhost:{port}",
                            server_id=server_name,
                            source="sidecar",
                        )
            except httpx.HTTPError:
                continue
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call(
        self,
        tool_name: str,
        arguments: dict | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict:
        """Dynamically invoke an MCP tool by name.

        Resolves the tool via the deployment-injected attached list,
        then the registry, then any legacy sidecars. On match, POSTs to
        ``{server_url}/execute`` (or ``/mcp/execute`` for legacy
        sidecars) with ``{tool, params}`` and returns the result.

        Raises:
            ToolNotFoundError: the tool is not accessible to this agent.
            ToolExecutionError: the tool call failed at transport level
                or the server returned an error body.
        """
        resolved = await self._resolve(tool_name)
        if resolved is None:
            raise ToolNotFoundError(
                f"Tool '{tool_name}' is not attached to this agent and "
                "could not be resolved via the registry or sidecar list."
            )

        args = arguments or {}
        # Merge any deployment-time default config underneath caller args.
        if resolved.config:
            merged: dict[str, Any] = {**resolved.config, **args}
        else:
            merged = args

        endpoint = self._execute_endpoint(resolved)
        payload = {"tool": tool_name, "params": merged}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise ToolExecutionError(
                f"Tool '{tool_name}' transport error: {exc}",
                tool_name=tool_name,
                server_url=resolved.server_url,
                cause=exc,
            ) from exc

        if resp.status_code >= 400:
            raise ToolExecutionError(
                f"Tool '{tool_name}' returned HTTP {resp.status_code}: "
                f"{resp.text[:200]}",
                tool_name=tool_name,
                server_url=resolved.server_url,
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise ToolExecutionError(
                f"Tool '{tool_name}' returned non-JSON body",
                tool_name=tool_name,
                server_url=resolved.server_url,
                cause=exc,
            ) from exc

        if isinstance(body, dict) and body.get("error"):
            raise ToolExecutionError(
                f"Tool '{tool_name}' reported error: {body['error']}",
                tool_name=tool_name,
                server_url=resolved.server_url,
            )

        # Most MCP servers wrap results in ``{"result": ...}``; unwrap
        # when present but otherwise return the full body so callers
        # with custom contracts still see everything.
        if isinstance(body, dict) and "result" in body:
            result = body["result"]
            if isinstance(result, dict):
                return result
            return {"result": result}
        if isinstance(body, dict):
            return body
        return {"result": body}

    @staticmethod
    def _execute_endpoint(resolved: _ResolvedTool) -> str:
        """Pick the right execute URL for the resolved tool.

        Legacy sidecars expose ``/mcp/execute``; newer servers register
        a plain ``/execute`` route. Attached/discovered tools use the
        server_url verbatim with ``/execute`` appended, which matches
        the MCP server contract documented in the SDK guide.
        """
        base = resolved.server_url.rstrip("/")
        if resolved.source == "sidecar":
            return f"{base}/mcp/execute"
        # Strip a trailing ``/mcp`` (present on some registry URLs)
        # before appending ``/execute`` so we always hit the action
        # handler rather than the tool listing endpoint.
        if base.endswith("/mcp"):
            base = base[: -len("/mcp")]
        return f"{base}/execute"

    # ------------------------------------------------------------------
    # Legacy discovery / execute API — preserved for existing callers.
    # ------------------------------------------------------------------

    async def discover(self) -> list[dict]:
        """Discover available tools via MCP token from the registry."""
        if self._token and self._registry_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{self._registry_url.rstrip('/')}/mcp/tools/discover",
                        headers={"X-MCP-Token": self._token},
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", [])
                        self._discovered_tools = data
                        self._refresh_registry_cache(data)
                        return data
            except httpx.HTTPError:
                pass
        return await self.list_tools()

    async def list_tools(self, server: str | None = None) -> list[dict]:
        """List available tools from MCP sidecars."""
        all_tools: list[dict] = []
        servers = (
            {server: self._servers[server]}
            if server and server in self._servers
            else self._servers
        )
        for name, port in servers.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"http://localhost:{port}/mcp")
                    if resp.status_code == 200:
                        tools = resp.json().get("tools", [])
                        for t in tools:
                            t["server"] = name
                        all_tools.extend(tools)
            except httpx.HTTPError:
                pass
        return all_tools

    async def execute(
        self,
        tool_name: str,
        params: dict | None = None,
        server: str | None = None,
    ) -> dict:
        """Execute an MCP tool on a sidecar or discovered server.

        Preserved for existing agents. New code should prefer
        ``ctx.tool(name, args)`` which wraps ``ToolsClient.call`` with
        a tracer span.
        """
        if params is None:
            params = {}

        # Prefer the new resolution path so attached tools just work.
        try:
            return await self.call(tool_name, params)
        except ToolNotFoundError:
            pass
        except ToolExecutionError:
            raise

        # Fall back to the legacy sidecar loop with an explicit server.
        port: int | None = None
        if server and server in self._servers:
            port = self._servers[server]
        else:
            for _name, p in self._servers.items():
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        resp = await client.get(f"http://localhost:{p}/mcp")
                        tools = resp.json().get("tools", [])
                        if any(t["name"] == tool_name for t in tools):
                            port = p
                            break
                except httpx.HTTPError:
                    continue

        if port is None:
            raise RuntimeError(
                f"Tool '{tool_name}' not found in any MCP sidecar or "
                "discovered server"
            )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"http://localhost:{port}/mcp/execute",
                json={"tool": tool_name, "params": params},
            )
            resp.raise_for_status()
            return resp.json().get("result", resp.json())
