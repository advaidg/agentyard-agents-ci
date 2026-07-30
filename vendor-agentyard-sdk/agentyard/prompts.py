"""Prompt library client for agents.

Usage::

    @yard.agent(name="invoice-extractor")
    async def extract(input, ctx):
        prompt = await ctx.prompts.render(
            "invoice-extraction",
            variables={"text": input["doc"]},
        )
        result = await call_llm(prompt.text)
        await ctx.prompts.record_usage(
            prompt.id,
            prompt.version,
            latency_ms=150,
            success=True,
            tokens_in=1200,
            tokens_out=350,
            cost_usd=0.012,
        )
        return result
"""

import os
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class RenderedPrompt:
    """Result of rendering a prompt template.

    ``id`` is the prompt UUID (as a string), ``version`` is the concrete
    version that was rendered, and ``text`` is the fully rendered output
    that can be passed to an LLM.
    """

    id: str
    version: int
    text: str
    variables_used: dict[str, Any] = field(default_factory=dict)


class PromptsClient:
    """Thin HTTP client that talks to the AgentYard registry prompt API.

    The client always hits the registry service directly (not the gateway)
    from inside the cluster, so it reuses ``AGENTYARD_REGISTRY_URL`` which
    is already set by the deployment generator for every agent pod.
    """

    def __init__(self, registry_url: str = "", namespace: str = "default"):
        self.registry_url = (
            registry_url
            or os.environ.get("AGENTYARD_REGISTRY_URL", "http://registry:8001")
        ).rstrip("/")
        self.namespace = namespace
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    @staticmethod
    def _unwrap(payload: dict[str, Any]) -> Any:
        """Unwrap the AgentYard envelope ``{ data, error }``."""
        if not isinstance(payload, dict):
            return payload
        if payload.get("error"):
            message = payload["error"].get("message", "unknown error")
            raise PromptClientError(message)
        return payload.get("data", payload)

    async def _get_prompt_id(self, name: str, namespace: str | None) -> str:
        ns = namespace or self.namespace
        client = await self._get_client()
        resp = await client.get(
            f"{self.registry_url}/prompts/by-name/{name}",
            params={"namespace": ns},
        )
        resp.raise_for_status()
        data = self._unwrap(resp.json())
        prompt_id = data.get("id") if isinstance(data, dict) else None
        if not prompt_id:
            raise PromptClientError(
                f"Prompt '{name}' not found in namespace '{ns}'"
            )
        return str(prompt_id)

    async def render(
        self,
        name: str,
        variables: dict[str, Any] | None = None,
        version: int | None = None,
        namespace: str | None = None,
    ) -> RenderedPrompt:
        """Resolve a prompt by name and render it server-side.

        Raises :class:`PromptClientError` on any HTTP or template error.
        """
        prompt_id = await self._get_prompt_id(name, namespace)
        client = await self._get_client()
        body: dict[str, Any] = {"variables": variables or {}}
        if version is not None:
            body["version"] = version
        resp = await client.post(
            f"{self.registry_url}/prompts/{prompt_id}/render",
            json=body,
        )
        if resp.status_code >= 400:
            raise PromptClientError(
                f"Prompt render failed ({resp.status_code}): {resp.text}"
            )
        data = self._unwrap(resp.json())
        if not isinstance(data, dict):
            raise PromptClientError("Unexpected render response shape")
        return RenderedPrompt(
            id=str(data.get("prompt_id", prompt_id)),
            version=int(data.get("version", version or 1)),
            text=str(data.get("rendered", "")),
            variables_used=dict(data.get("variables_used") or {}),
        )

    async def record_usage(
        self,
        prompt_id: str,
        version: int,
        latency_ms: int,
        success: bool,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        agent_name: str | None = None,
        invocation_id: str | None = None,
    ) -> None:
        """Record a single usage event. Failures are swallowed silently so
        agents never crash on telemetry errors.
        """
        client = await self._get_client()
        body: dict[str, Any] = {
            "version": version,
            "latency_ms": latency_ms,
            "success": success,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
        }
        if agent_name is not None:
            body["agent_name"] = agent_name
        if invocation_id is not None:
            body["invocation_id"] = invocation_id
        try:
            await client.post(
                f"{self.registry_url}/prompts/{prompt_id}/usage",
                json=body,
            )
        except httpx.HTTPError:
            # Usage recording must never break the agent.
            return

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class PromptClientError(Exception):
    """Raised when the prompt registry call fails."""
