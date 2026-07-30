"""Agent runtime context — the 3-verb surface every agent uses.

Design intent: an agent author needs to learn three nouns. That's the entire
SDK API. Capability stays the same; surface area drops.

    ctx.memory  — the unified "things I remember" surface. Routes to scratch,
                  episodic, semantic, or conversation backends based on how
                  you call it. One mental model, four storage strategies.
    ctx.llm     — generation. Always callable, always supports structured
                  output via ``schema=``. No JSON parsing in agents.
    ctx.invoke  — composition. Call another agent, system, capability, or
                  human. Same call shape, four kinds of dispatch.

Anything else you need (tracing, retry, secrets, raw HTTP) lives under
``ctx.advanced`` so it doesn't clutter day-to-day code.

Backwards compatible: every method on the legacy classes (``Memory``,
``LLMClient``, ``SecretClient``) keeps working. New code should use the new
verbs; old code keeps running unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Type

import httpx
from pydantic import BaseModel

from agentyard.v2.config import RuntimeConfig

logger = logging.getLogger("yard.context")


class MemoryAccessError(Exception):
    """Raised when an agent reads/writes a memory key not in its contract."""


class ResourceNotAvailable(Exception):
    """Raised when an agent accesses a resource it didn't declare needing."""


# ─── The unified Memory surface ─────────────────────────────────────────


@dataclass(frozen=True)
class MemoryHit:
    """One result from ``ctx.memory.find()``.

    Carries enough provenance for ``ctx.memory.note(out, sources=...)`` to
    record which hits informed the output.
    """

    id: str
    value: Any
    score: float = 1.0
    scope: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryFindResult:
    """Result of ``ctx.memory.find()`` — iterable list with extra accessors."""

    hits: list[MemoryHit] = field(default_factory=list)

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __bool__(self) -> bool:
        return bool(self.hits)

    @property
    def ids(self) -> list[str]:
        """For passing into ``ctx.memory.note(out, sources=hits.ids)``."""
        return [h.id for h in self.hits]

    @property
    def values(self) -> list[Any]:
        return [h.value for h in self.hits]


class Memory:
    """The single ``ctx.memory`` surface — routes by parameters.

    Four storage strategies under one API:

    - **scratch** (per-invocation): in-process dict. ``ctx.memory.scratch[k]``
    - **episodic** (per-system, persistent): Redis hash. ``ctx.memory.put/get``
    - **semantic** (vector recall): embedding-backed. ``ctx.memory.find(like=...)``
    - **conversation** (multi-turn thread): Redis list. ``ctx.memory.thread()``

    Backwards compatibility: the legacy ``ctx.memory[key] = value`` /
    ``ctx.memory[key]`` dict semantics still work and route to the episodic
    backend with ACL enforcement.
    """

    def __init__(
        self,
        contract: dict,
        redis_client=None,
        system_id: str = "",
        invocation_id: str = "",
    ):
        self._reads = set(contract.get("reads", []))
        self._writes = set(contract.get("writes", []))
        self._strict = contract.get("strict", False)
        self._redis = redis_client
        self._system_id = system_id
        self._invocation_id = invocation_id
        self._scratch: dict[str, Any] = {}
        # Per-invocation citation buffer; collected by the runtime envelope.
        self._citations: list[dict] = []

    # ── Scratch (per-invocation, never persisted) ───────────────────────

    @property
    def scratch(self) -> dict[str, Any]:
        """Per-invocation working memory. Cleared after the handler returns.

        Use this for intermediate values you don't need to remember after
        this invocation completes.
        """
        return self._scratch

    # ── Episodic (per-system, persistent k/v) ───────────────────────────

    async def put(
        self,
        key: str,
        value: Any,
        *,
        scope: str = "system",
        ttl: str | int | None = None,
    ) -> None:
        """Persist a value. Default scope is per-system.

        Args:
            key: Identifier (must satisfy the agent's writes contract).
            value: JSON-serializable value.
            scope: ``"system"`` (default) | ``"namespace"`` | ``"global"`` |
                custom string like ``"user:42"``.
            ttl: Seconds (int) or human string like ``"7d"``, ``"1h"``.
        """
        self._check_write(key)
        if not self._redis:
            self._scratch[f"{scope}:{key}"] = value
            return
        redis_key = self._scope_key(scope)
        payload = json.dumps(value, default=str)
        try:
            await self._redis.hset(redis_key, key, payload)
            if ttl is not None:
                seconds = _ttl_to_seconds(ttl)
                if seconds:
                    await self._redis.expire(redis_key, seconds)
        except Exception as exc:
            logger.warning("memory_put_failed key=%s err=%s", key, exc)

    async def get(self, key: str, *, scope: str = "system", default: Any = None) -> Any:
        """Read a value. Returns default when missing."""
        self._check_read(key)
        if not self._redis:
            return self._scratch.get(f"{scope}:{key}", default)
        redis_key = self._scope_key(scope)
        try:
            raw = await self._redis.hget(redis_key, key)
            if raw is None:
                return default
            return json.loads(raw)
        except Exception as exc:
            logger.warning("memory_get_failed key=%s err=%s", key, exc)
            return default

    # ── Semantic (vector-indexed recall) ────────────────────────────────

    async def find(
        self,
        *,
        like: str | None = None,
        key: str | None = None,
        scope: str = "system",
        k: int = 5,
        min_score: float = 0.05,
    ) -> MemoryFindResult:
        """Recall similar items. Either ``like=text`` (semantic) or
        ``key=...`` (exact key lookup as a degenerate single-hit search).

        Returns a ``MemoryFindResult`` whose ``.ids`` you can pass to
        :meth:`note` as the ``sources=`` arg for automatic provenance.

        Backend strategy: token-overlap with bigram shingling and Jaccard-
        style scoring. Stdlib-only, no numpy / vector DB needed for v1.
        Same API contract when an embedding backend is plugged in later
        (set ``YARD_EMBEDDINGS=anthropic`` or similar).

        Args:
            like: Free-text query. Returns items whose stored values contain
                similar tokens, ranked high-to-low.
            key: Exact key lookup, returns at most one hit. Mutually
                exclusive with ``like``.
            scope: Memory scope to search (defaults to ``"system"``).
            k: Max hits to return.
            min_score: Hits below this threshold are dropped.
        """
        if key is not None:
            value = await self.get(key, scope=scope)
            if value is None:
                return MemoryFindResult(hits=[])
            return MemoryFindResult(
                hits=[MemoryHit(id=key, value=value, score=1.0, scope=scope)]
            )

        if like is None or not self._redis:
            return MemoryFindResult(hits=[])

        redis_key = self._scope_key(scope)
        try:
            entries = await self._redis.hgetall(redis_key)
        except Exception as exc:
            logger.warning("memory_find_failed scope=%s err=%s", scope, exc)
            return MemoryFindResult(hits=[])

        query_shingles = _shingle(like)
        if not query_shingles:
            return MemoryFindResult(hits=[])

        scored: list[tuple[float, MemoryHit]] = []
        for entry_key, raw in entries.items():
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            text = str(value)
            doc_shingles = _shingle(text)
            score = _jaccard(query_shingles, doc_shingles)
            if score >= min_score:
                scored.append((score, MemoryHit(
                    id=entry_key, value=value, score=score, scope=scope,
                )))
        scored.sort(key=lambda t: t[0], reverse=True)
        return MemoryFindResult(hits=[h for _, h in scored[:k]])

    async def note(
        self,
        value: Any,
        *,
        key: str | None = None,
        scope: str = "system",
        sources: list[str] | None = None,
    ) -> None:
        """Persist + cite in one call. Equivalent to ``put`` + ``cite``.

        ``sources`` is a list of memory ids that informed this value (e.g.
        the ``.ids`` from a prior ``find()``). They get attached to the
        invocation envelope automatically.
        """
        if key is None:
            # Auto-derive a key when none given — invocation-scoped.
            key = f"{self._invocation_id or 'note'}:{len(self._scratch)}"
        await self.put(key, value, scope=scope)
        if sources:
            self.cite(*sources)

    # ── Conversation (multi-turn thread) ────────────────────────────────

    def thread(
        self,
        *,
        scope: str | None = None,
        max_turns: int = 50,
    ) -> "ConversationThread":
        """Return the conversation thread for this scope.

        Default scope is the invocation_id (per-invocation conversation).
        Pass a stable scope string (e.g. ``"user:42"``) for cross-invocation
        threads.
        """
        return ConversationThread(
            redis=self._redis,
            scope=scope or self._invocation_id or "default",
            max_turns=max_turns,
            system_id=self._system_id,
        )

    # ── Provenance / citations ──────────────────────────────────────────

    def cite(self, *source_ids: str, **metadata: Any) -> None:
        """Record that one or more memory ids contributed to this output.

        Citations are picked up by the runtime envelope and shipped with
        the result, so downstream consumers (or the audit stream) know
        what informed it.
        """
        for sid in source_ids:
            self._citations.append({"source": sid, **metadata})

    def consume_citations(self) -> list[dict]:
        """Internal: runtime calls this to drain + attach to the envelope."""
        out = list(self._citations)
        self._citations.clear()
        return out

    # ── Backwards-compatible dict semantics ─────────────────────────────

    def __getitem__(self, key: str) -> Any:
        """Legacy: ``ctx.memory[k]`` reads from episodic scope synchronously."""
        self._check_read(key)
        if self._redis and self._system_id:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Inside an async context; can't block. Fall back to scratch.
                    return self._scratch.get(f"system:{key}")
                val = loop.run_until_complete(
                    self._redis.hget(self._scope_key("system"), key)
                )
                if val:
                    return json.loads(val)
            except Exception:
                pass
        return self._scratch.get(f"system:{key}")

    def __setitem__(self, key: str, value: Any) -> None:
        """Legacy: ``ctx.memory[k] = v`` — async fire-and-forget."""
        self._check_write(key)
        self._scratch[f"system:{key}"] = value
        if self._redis and self._system_id:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(
                        self._redis.hset(
                            self._scope_key("system"),
                            key,
                            json.dumps(value, default=str),
                        )
                    )
                else:
                    loop.run_until_complete(
                        self._redis.hset(
                            self._scope_key("system"),
                            key,
                            json.dumps(value, default=str),
                        )
                    )
            except Exception as exc:
                logger.warning("memory_persist_failed: %s", exc)

    # ── Internals ───────────────────────────────────────────────────────

    def _check_read(self, key: str) -> None:
        if self._reads and key not in self._reads:
            msg = f"Memory read denied: {key!r} not in declared reads {sorted(self._reads)}"
            if self._strict:
                raise MemoryAccessError(msg)
            logger.warning("memory_acl_violation: %s", msg)

    def _check_write(self, key: str) -> None:
        if self._writes and key not in self._writes:
            msg = f"Memory write denied: {key!r} not in declared writes {sorted(self._writes)}"
            if self._strict:
                raise MemoryAccessError(msg)
            logger.warning("memory_acl_violation: %s", msg)

    def _scope_key(self, scope: str) -> str:
        if scope == "global":
            return "yard:memory:global"
        if scope == "namespace":
            ns = (self._system_id or "default").split("/")[0]
            return f"yard:memory:ns:{ns}"
        if scope == "system":
            return f"yard:system:{self._system_id}:memory"
        # custom scope (e.g. "user:42") — namespace under the system
        return f"yard:system:{self._system_id}:scope:{scope}"


class ConversationThread:
    """Multi-turn conversation history with token-aware windowing.

    Append turns as you go; ``window()`` returns the most recent N turns
    that fit within a token budget. Used for chat-style agents that need
    to remember what was said earlier in the same (or related) invocation.
    """

    def __init__(self, redis, scope: str, max_turns: int, system_id: str):
        self._redis = redis
        self._scope = scope
        self._max_turns = max_turns
        self._system_id = system_id
        self._key = f"yard:system:{system_id}:thread:{scope}"
        self._buffer: list[dict] = []  # in-memory buffer when no redis

    async def append(self, *, role: str, content: str, **metadata: Any) -> None:
        """Add one turn to the conversation."""
        turn = {"role": role, "content": content, **metadata}
        if not self._redis:
            self._buffer.append(turn)
            self._buffer = self._buffer[-self._max_turns :]
            return
        try:
            await self._redis.rpush(self._key, json.dumps(turn, default=str))
            await self._redis.ltrim(self._key, -self._max_turns, -1)
        except Exception as exc:
            logger.warning("conversation_append_failed scope=%s err=%s", self._scope, exc)
            self._buffer.append(turn)

    async def window(self, *, max_turns: int | None = None, max_tokens: int | None = None) -> list[dict]:
        """Return the most recent turns, optionally bounded by token count.

        Token counting is conservative (4 chars ≈ 1 token). Real tokenizer
        integration is a non-goal at the SDK level — agents that need exact
        counts should use their LLM provider's tokenizer.
        """
        n = max_turns or self._max_turns
        turns: list[dict] = []
        if self._redis:
            try:
                raw_list = await self._redis.lrange(self._key, -n, -1)
                turns = [json.loads(r) for r in raw_list]
            except Exception as exc:
                logger.warning("conversation_window_failed scope=%s err=%s", self._scope, exc)
                turns = list(self._buffer[-n:])
        else:
            turns = list(self._buffer[-n:])

        if max_tokens is None:
            return turns
        # Truncate from the front to fit token budget
        kept: list[dict] = []
        running = 0
        for turn in reversed(turns):
            est = len(str(turn.get("content", ""))) // 4 + 8
            if running + est > max_tokens:
                break
            kept.append(turn)
            running += est
        return list(reversed(kept))

    async def clear(self) -> None:
        if self._redis:
            try:
                await self._redis.delete(self._key)
            except Exception:
                pass
        self._buffer.clear()


# ─── The unified LLM surface ────────────────────────────────────────────


# Sensible default model per provider. Overridable via config["model"] (set
# by the topology compiler's /yard/config.yaml) or the YARD_LLM_MODEL env
# var (used when no config.yaml exists, e.g. plain docker-compose agents).
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    # Bedrock requires the cross-region inference profile ID for this model,
    # not the bare on-demand foundation-model ID — Bedrock rejects on-demand
    # invocation for it with "isn't supported ... use an inference profile".
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "openai": "gpt-4o-mini",
    "cohere": "command-r",
}

# Provider -> the env var holding its API key. Bedrock has none here — it
# authenticates via the ambient AWS credential chain (instance role, env
# creds, etc.), not a single static key.
_PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
}


class LLM:
    """Callable LLM surface.

    Two ways to call:
      result = await ctx.llm("prompt")              # returns str
      result = await ctx.llm("prompt", schema=Out)  # returns Out instance

    The structured-output path validates against the Pydantic schema before
    returning. Failed validation surfaces a clear error rather than a
    silently-malformed result.

    Provider is pluggable (anthropic / bedrock / openai / cohere), resolved
    from config["provider"] when a /yard/config.yaml exists (K8s/mesh
    deploys), falling back to YARD_LLM_PROVIDER for agents that run without
    one (e.g. plain docker-compose) — same pattern as the api_key fallback
    below. Defaults to "anthropic" for backwards compatibility.
    """

    def __init__(self, config: dict):
        self.provider = config.get("provider") or os.environ.get("YARD_LLM_PROVIDER", "anthropic")
        self.model = (
            config.get("model")
            or os.environ.get("YARD_LLM_MODEL")
            or _DEFAULT_MODELS.get(self.provider, _DEFAULT_MODELS["anthropic"])
        )
        env_key = _PROVIDER_API_KEY_ENV.get(self.provider)
        self.api_key = config.get("api_key") or (os.environ.get(env_key, "") if env_key else "")
        self.bedrock_region = (
            config.get("bedrock_region")
            or os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or "us-east-1"
        )

    async def __call__(
        self,
        prompt: str,
        *,
        schema: Type[BaseModel] | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Any:
        """Generate. With ``schema=``, returns a parsed Pydantic instance."""
        full_prompt = prompt
        if schema is not None:
            schema_text = json.dumps(schema.model_json_schema(), indent=2)
            full_prompt = (
                f"{prompt}\n\n"
                f"Respond with ONLY a JSON object matching this schema. "
                f"No prose, no markdown fences:\n{schema_text}"
            )
        text = await self.complete(
            full_prompt, model=model, max_tokens=max_tokens, temperature=temperature,
        )
        if schema is None:
            return text
        # Strip code fences if present (LLMs often add them despite instructions)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON: {text[:200]!r}") from exc
        return schema(**data)

    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Plain text generation. Same as ``ctx.llm(prompt)`` without ``schema=``."""
        use_model = model or self.model
        if self.provider == "anthropic":
            return await self._complete_anthropic(prompt, use_model, max_tokens, temperature)
        if self.provider == "bedrock":
            return await self._complete_bedrock(prompt, use_model, max_tokens, temperature)
        if self.provider == "openai":
            return await self._complete_openai(prompt, use_model, max_tokens, temperature)
        if self.provider == "cohere":
            return await self._complete_cohere(prompt, use_model, max_tokens, temperature)
        raise NotImplementedError(f"LLM provider {self.provider!r} not implemented")

    async def _complete_anthropic(
        self, prompt: str, model: str, max_tokens: int, temperature: float,
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            body = resp.json()
            return body["content"][0]["text"]

    async def _complete_bedrock(
        self, prompt: str, model: str, max_tokens: int, temperature: float,
    ) -> str:
        """Claude via AWS Bedrock. Authenticates via the ambient AWS
        credential chain (EC2/EKS instance role, env creds, etc.) — no
        static API key needed, matching Bedrock's own auth model.

        boto3 is synchronous, so the actual call runs in a thread to avoid
        blocking the event loop other requests share.
        """
        import boto3  # local import: optional dep, only needed for bedrock

        def _invoke() -> str:
            client = boto3.client("bedrock-runtime", region_name=self.bedrock_region)
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            })
            resp = client.invoke_model(modelId=model, body=body)
            payload = json.loads(resp["body"].read())
            return payload["content"][0]["text"]

        return await asyncio.to_thread(_invoke)

    async def _complete_openai(
        self, prompt: str, model: str, max_tokens: int, temperature: float,
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            body = resp.json()
            return body["choices"][0]["message"]["content"]

    async def _complete_cohere(
        self, prompt: str, model: str, max_tokens: int, temperature: float,
    ) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.cohere.com/v2/chat",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            body = resp.json()
            return body["message"]["content"][0]["text"]


# Legacy alias — kept so existing agents using ctx.llm.complete() still work.
LLMClient = LLM


# ─── The unified invoke surface ─────────────────────────────────────────


class Invoke:
    """Composition primitive — call another agent / system / capability / human.

    Same call shape, four kinds of dispatch::

        await ctx.invoke(agent="acme/sentiment", input={...})
        await ctx.invoke(system="acme/refund-flow", input={...})
        await ctx.invoke(capability="document.summarize", input={...})
        await ctx.invoke(human="approver", prompt="Approve $1200?", schema={...})

    All four return the result; agent/system/capability return the agent's
    output, human returns the human's response (or raises on timeout).
    """

    def __init__(self, *, registry_url: str, gateway_url: str, system_id: str, invocation_id: str):
        self._registry_url = registry_url
        self._gateway_url = gateway_url
        self._system_id = system_id
        self._invocation_id = invocation_id

    async def __call__(
        self,
        *,
        agent: str | None = None,
        system: str | None = None,
        capability: str | None = None,
        human: str | None = None,
        input: Any = None,
        prompt: str | None = None,
        schema: Any = None,
        timeout_seconds: float = 60.0,
    ) -> Any:
        kinds = sum(x is not None for x in (agent, system, capability, human))
        if kinds != 1:
            raise ValueError(
                "ctx.invoke needs exactly one of: agent=, system=, capability=, human="
            )
        headers = {
            "X-Invocation-ID": self._invocation_id,
            "X-Trace-ID": self._invocation_id,
            "Content-Type": "application/json",
        }
        if agent:
            return await self._invoke_agent(agent, input, headers, timeout_seconds)
        if system:
            return await self._invoke_system(system, input, headers, timeout_seconds)
        if capability:
            return await self._invoke_capability(capability, input, headers, timeout_seconds)
        if human:
            return await self._invoke_human(human, prompt or "", schema, timeout_seconds)
        return None  # unreachable

    async def _invoke_agent(self, name: str, input: Any, headers: dict, timeout: float) -> Any:
        """Look up an agent by name in the registry, POST /invoke against it."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                lookup = await client.get(
                    f"{self._registry_url}/agents/by-name/{name}", timeout=5.0,
                )
                lookup.raise_for_status()
                a2a_url = lookup.json().get("data", {}).get("a2a_endpoint", "")
            except Exception as exc:
                raise RuntimeError(f"ctx.invoke(agent={name!r}) — registry lookup failed: {exc}")
            if not a2a_url:
                raise RuntimeError(f"agent {name!r} has no a2a_endpoint registered")
            resp = await client.post(
                f"{a2a_url.rstrip('/')}/invoke",
                json={"input": input}, headers=headers,
            )
            resp.raise_for_status()
            body = resp.json()
            return body.get("output", body)

    async def _invoke_system(self, system: str, input: Any, headers: dict, timeout: float) -> Any:
        """Trigger another full system as a sub-workflow."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{self._gateway_url}/api/systems/{system}/invoke",
                json={"input": input}, headers=headers,
            )
            resp.raise_for_status()
            return resp.json().get("data", resp.json())

    async def _invoke_capability(self, capability: str, input: Any, headers: dict, timeout: float) -> Any:
        """Find any registered agent that declares this capability and call it."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                lookup = await client.get(
                    f"{self._registry_url}/agents",
                    params={"capabilities": capability, "limit": 1},
                    timeout=5.0,
                )
                lookup.raise_for_status()
                items = lookup.json().get("data", {}).get("items", [])
            except Exception as exc:
                raise RuntimeError(f"capability lookup failed: {exc}")
            if not items:
                raise RuntimeError(f"no registered agent declares capability {capability!r}")
            agent_name = items[0].get("name")
            return await self._invoke_agent(agent_name, input, headers, timeout)

    async def _invoke_human(
        self,
        human_id: str,
        prompt: str,
        schema: Any,
        timeout_seconds: float,
    ) -> Any:
        """Pause the workflow waiting for a human response.

        v1 implementation uses Redis BLPOP on a per-request key. Mission
        Control's "Approvals" surface is the producer; until that's wired
        to ctx.invoke(human=...), the call will block until the timeout.
        """
        import uuid
        request_id = uuid.uuid4().hex
        # Push the request onto a Redis list the UI subscribes to, then
        # block waiting for the response. Redis BLPOP gives us the timeout
        # semantics for free.
        from redis.asyncio import from_url
        redis_url = os.environ.get("YARD_REDIS_URL") or "redis://redis:6379/0"
        rc = from_url(redis_url, decode_responses=True)
        request_key = f"yard:human:{self._system_id}:{human_id}:requests"
        response_key = f"yard:human:{self._system_id}:{human_id}:resp:{request_id}"
        try:
            await rc.rpush(
                request_key,
                json.dumps({
                    "id": request_id,
                    "invocation_id": self._invocation_id,
                    "prompt": prompt,
                    "schema": schema,
                    "ts": __import__("time").time(),
                }, default=str),
            )
            blocked = await rc.blpop(response_key, timeout=int(timeout_seconds))
            if blocked is None:
                raise TimeoutError(f"human {human_id!r} did not respond within {timeout_seconds}s")
            _, raw = blocked
            return json.loads(raw)
        finally:
            try:
                await rc.aclose()
            except Exception:
                pass


# ─── Secrets (kept simple; advanced surface) ────────────────────────────


class SecretClient:
    """Resolve secrets the agent declared needing."""

    def __init__(self, secrets: dict[str, str], declared: list[str]):
        self._secrets = secrets
        self._declared = set(declared)

    def __call__(self, name: str) -> str:
        if self._declared and name not in self._declared:
            raise ResourceNotAvailable(f"Secret {name!r} not declared in agent's needs")
        if name in self._secrets:
            return self._secrets[name]
        val = os.environ.get(name, "")
        if not val:
            raise ResourceNotAvailable(f"Secret {name!r} not available")
        return val


# ─── The 3-verb context object ──────────────────────────────────────────


class _Advanced:
    """Power-user surface — kept off the main API to keep ctx clean.

    Use ``ctx.advanced`` only when you need explicit control:
        ctx.advanced.fetch(url)         # raw HTTP GET
        ctx.advanced.span("step")       # OTLP tracing span
        ctx.advanced.breaker("name")    # circuit breaker
        ctx.advanced.emit("event", ...)  # publish to event bus
    """

    def __init__(self, parent: "AgentContext"):
        self._parent = parent

    async def fetch(self, url: str, **kwargs) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, **kwargs)
            resp.raise_for_status()
            return resp.text


class AgentContext:
    """The ``ctx`` argument every agent receives.

    Three nouns:
        ctx.memory   — recall + remember + cite, across 4 storage strategies
        ctx.llm      — generate (callable; ``schema=`` for structured output)
        ctx.invoke   — call another agent / system / capability / human

    Plus housekeeping:
        ctx.invocation_id, ctx.trace_id, ctx.system_id, ctx.node_id
        ctx.secret(name) — resolve a declared secret
        ctx.advanced — power-user escape hatch (tracer, fetch, breaker, emit)
    """

    def __init__(self, config: RuntimeConfig, agent_meta: dict):
        self.config = config
        self.agent_meta = agent_meta
        self.system_id = config.system_id
        self.node_id = config.node_id
        self.invocation_id = ""
        self.trace_id = ""

        # ── ctx.memory — unified surface ────────────────────────────────
        # Lazy-init a Redis client so the new put/get/find methods can
        # persist beyond per-process scratch. Failures here are fatal in
        # strict mode; otherwise scratch-only memory still works.
        memory_redis = None
        try:
            import redis.asyncio as _aioredis  # local import: optional dep
            memory_redis = _aioredis.from_url(
                config.redis_url, decode_responses=True,
            )
        except Exception as exc:
            logger.warning("memory_redis_unavailable err=%s — falling back to scratch", exc)
        self.memory = Memory(
            agent_meta.get("memory", {}),
            redis_client=memory_redis,
            system_id=self.system_id,
            invocation_id="",
        )

        # ── ctx.llm — callable ──────────────────────────────────────────
        self.llm: LLM | None = None
        for resource in agent_meta.get("needs", []):
            if hasattr(resource, "kind") and resource.kind.value == "llm":
                self.llm = LLM(config.resources.get("llm", {}))
                break

        # ── ctx.invoke — composition ────────────────────────────────────
        registry_url = os.environ.get("AGENTYARD_REGISTRY_URL", "http://registry:8001")
        gateway_url = os.environ.get("YARD_GATEWAY_URL", "http://gateway:8000")
        self.invoke = Invoke(
            registry_url=registry_url,
            gateway_url=gateway_url,
            system_id=self.system_id,
            invocation_id="",
        )

        # ── ctx.secret — declared resolver ──────────────────────────────
        declared_secrets: list[str] = []
        for resource in agent_meta.get("needs", []):
            if hasattr(resource, "kind") and resource.kind.value == "secrets":
                declared_secrets.extend(resource.options.get("keys", []))
        self.secret = SecretClient(config.secrets, declared_secrets)

        # ── ctx.advanced — escape hatch ─────────────────────────────────
        self.advanced = _Advanced(self)

    def _set_invocation(self, invocation_id: str, trace_id: str = "") -> None:
        """Runtime hook — called per-invocation to thread ids through ctx."""
        self.invocation_id = invocation_id
        self.trace_id = trace_id
        self.memory._invocation_id = invocation_id
        self.invoke._invocation_id = invocation_id

    # ── Backwards-compat shims ──────────────────────────────────────────

    async def fetch(self, url: str, **kwargs) -> str:
        """Deprecated alias for ``ctx.advanced.fetch``."""
        return await self.advanced.fetch(url, **kwargs)


# ─── helpers ─────────────────────────────────────────────────────────────


def _shingle(text: str, n: int = 3) -> set[str]:
    """Tokenize + bigram-shingle a string for Jaccard-style similarity.

    Lowercases, strips non-alphanumeric, builds char n-grams over each
    token. Robust to typos and word-order shifts. Stdlib only.
    """
    import re
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out: set[str] = set()
    for tok in tokens:
        if len(tok) <= n:
            out.add(tok)
        else:
            for i in range(len(tok) - n + 1):
                out.add(tok[i : i + n])
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity: |intersection| / |union|. 0.0 to 1.0."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _ttl_to_seconds(ttl: str | int | None) -> int | None:
    """Parse '7d', '1h', '30m', '60s', or an int into seconds."""
    if ttl is None:
        return None
    if isinstance(ttl, int):
        return ttl
    s = str(ttl).strip().lower()
    if not s:
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if s[-1] in units:
        try:
            return int(s[:-1]) * units[s[-1]]
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None
