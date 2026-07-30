"""LLM client for AgentYard agents.

Provides a unified interface across OpenAI, Anthropic, and AWS Bedrock.
Handles streaming, retry, token counting, semantic caching, and observability.

Usage:
    response = await ctx.llm.complete(
        prompt="Summarize this doc: ...",
        model="gpt-4o",
        max_tokens=500,
        temperature=0.2,
    )
    # response.text, response.tokens_in, response.tokens_out, response.cost_usd

    async for chunk in ctx.llm.stream(prompt="...", model="claude-3-5-sonnet"):
        print(chunk.delta, end="")
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

logger = logging.getLogger("agentyard.llm")

Provider = Literal["openai", "anthropic", "bedrock"]

# Model routing — maps model name → provider
MODEL_PROVIDERS: dict[str, Provider] = {
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4-turbo": "openai",
    "gpt-3.5-turbo": "openai",
    "claude-3-5-sonnet": "anthropic",
    "claude-3-5-haiku": "anthropic",
    "claude-3-opus": "anthropic",
    "claude-opus-4-6": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "claude-haiku-4-5": "anthropic",
}

# Cost per 1M tokens (USD). Rough as of 2025; keep simple to update.
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # (input_per_1m, output_per_1m)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: Provider
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    finish_reason: str = "stop"
    cached: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class LLMChunk:
    delta: str
    model: str
    finish_reason: str | None = None


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(Exception):
    """Base for LLM client errors."""


class LLMProviderError(LLMError):
    """The upstream provider returned an error."""


class LLMRateLimitError(LLMError):
    """Provider rate-limited us; caller may retry."""


class LLMClient:
    """Unified LLM client with provider routing, retry, caching, and metrics."""

    def __init__(
        self,
        *,
        default_model: str | None = None,
        redis_url: str = "",
        agent_name: str = "",
        cache_ttl_seconds: int = 300,
        max_retries: int = 3,
        default_timeout: float = 60.0,
    ):
        self.default_model = default_model or os.environ.get(
            "YARD_LLM_DEFAULT_MODEL", "gpt-4o-mini"
        )
        self.redis_url = redis_url or os.environ.get("YARD_REDIS_URL", "")
        self.agent_name = agent_name
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_retries = max_retries
        self.default_timeout = default_timeout

    def _resolve_provider(self, model: str) -> Provider:
        provider = MODEL_PROVIDERS.get(model)
        if provider is None:
            # Heuristic fallback
            if model.startswith("gpt-") or model.startswith("o1-"):
                return "openai"
            if model.startswith("claude"):
                return "anthropic"
            return "openai"
        return provider

    def _estimate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        costs = MODEL_COSTS.get(model, (0.0, 0.0))
        return (tokens_in / 1_000_000) * costs[0] + (tokens_out / 1_000_000) * costs[1]

    def _cache_key(
        self, prompt: str, model: str, temperature: float, max_tokens: int | None
    ) -> str:
        key_src = f"{model}|{temperature}|{max_tokens}|{prompt}"
        return f"yard:llm:cache:{hashlib.sha256(key_src.encode()).hexdigest()[:32]}"

    async def _cache_get(self, key: str) -> LLMResponse | None:
        if not self.redis_url:
            return None
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(self.redis_url, decode_responses=True)
            try:
                raw = await r.get(key)
                if not raw:
                    return None
                data = json.loads(raw)
                return LLMResponse(**{**data, "cached": True})
            finally:
                await r.aclose()
        except Exception as e:
            logger.debug(f"llm cache get failed: {e}")
            return None

    async def _cache_set(self, key: str, response: LLMResponse) -> None:
        if not self.redis_url:
            return
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(self.redis_url, decode_responses=True)
            try:
                data = {
                    "text": response.text,
                    "model": response.model,
                    "provider": response.provider,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "finish_reason": response.finish_reason,
                    "raw": response.raw,
                }
                await r.setex(key, self.cache_ttl_seconds, json.dumps(data))
            finally:
                await r.aclose()
        except Exception as e:
            logger.debug(f"llm cache set failed: {e}")

    async def complete(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: str | None = None,
        use_cache: bool = True,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Run an LLM completion."""
        model = model or self.default_model
        provider = self._resolve_provider(model)
        prompt_str = (
            prompt
            if isinstance(prompt, str)
            else json.dumps(
                [{"role": m.role, "content": m.content} for m in prompt]
            )
        )

        cache_key = ""
        if use_cache:
            cache_key = self._cache_key(prompt_str, model, temperature, max_tokens)
            cached = await self._cache_get(cache_key)
            if cached:
                return cached

        start = time.monotonic()
        response = await self._call_provider(
            provider,
            prompt,
            model,
            max_tokens,
            temperature,
            system,
            timeout or self.default_timeout,
        )
        response.latency_ms = int((time.monotonic() - start) * 1000)

        if use_cache:
            await self._cache_set(cache_key, response)

        return response

    async def _call_provider(
        self, provider: Provider, prompt, model, max_tokens, temperature, system, timeout
    ) -> LLMResponse:
        import httpx

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if provider == "openai":
                    return await self._call_openai(
                        prompt, model, max_tokens, temperature, system, timeout
                    )
                if provider == "anthropic":
                    return await self._call_anthropic(
                        prompt, model, max_tokens, temperature, system, timeout
                    )
                if provider == "bedrock":
                    return await self._call_bedrock(
                        prompt, model, max_tokens, temperature, system, timeout
                    )
                raise LLMError(f"Unknown provider: {provider}")
            except LLMRateLimitError as e:
                last_err = e
                await asyncio.sleep(min(2**attempt, 30))
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_err = LLMProviderError(f"Network error: {e}")
                await asyncio.sleep(min(2**attempt, 30))
        raise last_err or LLMError("Unknown LLM error after retries")

    async def _call_openai(
        self, prompt, model, max_tokens, temperature, system, timeout
    ) -> LLMResponse:
        import httpx

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMError("OPENAI_API_KEY not set")

        messages = self._to_messages(prompt, system)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            if resp.status_code == 429:
                raise LLMRateLimitError(resp.text)
            if resp.status_code >= 400:
                raise LLMProviderError(f"OpenAI {resp.status_code}: {resp.text}")
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            return LLMResponse(
                text=choice["message"]["content"],
                model=model,
                provider="openai",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=self._estimate_cost(model, tokens_in, tokens_out),
                latency_ms=0,
                finish_reason=choice.get("finish_reason", "stop"),
                raw=data,
            )

    async def _call_anthropic(
        self, prompt, model, max_tokens, temperature, system, timeout
    ) -> LLMResponse:
        import httpx

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")

        messages = [m for m in self._to_messages(prompt, None) if m["role"] != "system"]
        sys_text = system or ""
        if isinstance(prompt, list):
            for m in prompt:
                if m.role == "system":
                    sys_text = m.content if not sys_text else f"{sys_text}\n\n{m.content}"

        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if sys_text:
            body["system"] = sys_text

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            if resp.status_code == 429:
                raise LLMRateLimitError(resp.text)
            if resp.status_code >= 400:
                raise LLMProviderError(f"Anthropic {resp.status_code}: {resp.text}")
            data = resp.json()
            content_blocks = data.get("content", [])
            text = "".join(
                b.get("text", "") for b in content_blocks if b.get("type") == "text"
            )
            usage = data.get("usage", {})
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            return LLMResponse(
                text=text,
                model=model,
                provider="anthropic",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=self._estimate_cost(model, tokens_in, tokens_out),
                latency_ms=0,
                finish_reason=data.get("stop_reason", "stop"),
                raw=data,
            )

    async def _call_bedrock(
        self, prompt, model, max_tokens, temperature, system, timeout
    ) -> LLMResponse:
        # Stub: Bedrock requires boto3 + SigV4. Document as TODO for future.
        raise LLMError(
            "Bedrock provider is not implemented yet — use openai or anthropic"
        )

    async def stream(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: str | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream LLM tokens as they arrive."""
        model = model or self.default_model
        provider = self._resolve_provider(model)
        if provider == "openai":
            async for chunk in self._stream_openai(
                prompt,
                model,
                max_tokens,
                temperature,
                system,
                timeout or self.default_timeout,
            ):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._stream_anthropic(
                prompt,
                model,
                max_tokens,
                temperature,
                system,
                timeout or self.default_timeout,
            ):
                yield chunk
        else:
            raise LLMError(f"Streaming not supported for provider: {provider}")

    async def _stream_openai(
        self, prompt, model, max_tokens, temperature, system, timeout
    ):
        import httpx

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMError("OPENAI_API_KEY not set")
        messages = self._to_messages(prompt, system)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise LLMProviderError(
                        f"OpenAI {resp.status_code}: {body.decode()}"
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        return
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choice = (data.get("choices") or [{}])[0]
                    delta = choice.get("delta", {}).get("content") or ""
                    finish = choice.get("finish_reason")
                    if delta or finish:
                        yield LLMChunk(delta=delta, model=model, finish_reason=finish)

    async def _stream_anthropic(
        self, prompt, model, max_tokens, temperature, system, timeout
    ):
        import httpx

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        messages = [m for m in self._to_messages(prompt, None) if m["role"] != "system"]
        sys_text = system or ""
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
        }
        if sys_text:
            body["system"] = sys_text
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            ) as resp:
                if resp.status_code >= 400:
                    body_bytes = await resp.aread()
                    raise LLMProviderError(
                        f"Anthropic {resp.status_code}: {body_bytes.decode()}"
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    evt_type = data.get("type", "")
                    if evt_type == "content_block_delta":
                        delta = data.get("delta", {}).get("text", "")
                        if delta:
                            yield LLMChunk(delta=delta, model=model)
                    elif evt_type == "message_stop":
                        yield LLMChunk(delta="", model=model, finish_reason="stop")
                        return

    def _to_messages(self, prompt, system: str | None) -> list[dict]:
        if isinstance(prompt, str):
            messages: list[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return messages
        return [{"role": m.role, "content": m.content} for m in prompt]
