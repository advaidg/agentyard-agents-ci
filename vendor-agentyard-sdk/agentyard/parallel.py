"""Parallel primitives for agent orchestration.

These primitives let one agent fan out work to other agents with bounded
concurrency, fold results through a reducer agent, race multiple agents
against each other, or gather a fixed list of calls.

All primitives delegate to ``ctx.call(agent_name, input)`` so they
inherit the circuit breaker and trace propagation that the SDK already
provides for agent-to-agent calls.

Usage:
    @yard.agent(name="bulk-classifier")
    async def classify(input, ctx):
        result = await ctx.map(
            "single-classifier",
            input["docs"],
            concurrency=20,
            timeout=10.0,
        )
        return {"labels": result.successes, "failed": len(result.failures)}
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("agentyard.parallel")


@dataclass
class MapResult:
    """Output of :meth:`ParallelPrimitives.map`.

    ``successes`` is positional — index ``i`` corresponds to ``items[i]``,
    or ``None`` if that index failed. ``failures`` is a list of
    ``(index, error_message)`` tuples for the failed items.
    """

    successes: list[Any]
    failures: list[tuple[int, str]]
    duration_ms: int

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.successes if s is not None)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


class ParallelPrimitives:
    """Implementation of ctx.map / ctx.reduce / ctx.race / ctx.gather."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

    async def map(
        self,
        agent_name: str,
        items: list[Any],
        *,
        concurrency: int = 10,
        timeout: float = 60.0,
        fail_fast: bool = False,
    ) -> MapResult:
        """Call ``agent_name`` once per item with bounded concurrency."""
        start = time.monotonic()
        sem = asyncio.Semaphore(max(1, concurrency))
        successes: list[Any] = [None] * len(items)
        failures: list[tuple[int, str]] = []
        first_error: Exception | None = None

        async def call_one(idx: int, item: Any) -> None:
            nonlocal first_error
            async with sem:
                try:
                    result = await asyncio.wait_for(
                        self.ctx.call(agent_name, item),
                        timeout=timeout,
                    )
                    successes[idx] = result
                except Exception as exc:
                    failures.append((idx, str(exc)))
                    if fail_fast and first_error is None:
                        first_error = exc

        tasks = [
            asyncio.create_task(call_one(i, item)) for i, item in enumerate(items)
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if fail_fast and first_error is not None:
            raise first_error

        return MapResult(
            successes=successes,
            failures=failures,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def reduce(
        self,
        reducer_agent: str,
        items: list[Any],
        *,
        initial: Any = None,
        chunk_size: int = 10,
    ) -> Any:
        """Fold ``items`` through a reducer agent.

        The reducer receives ``{"accumulator": ..., "chunk": [...]}`` and
        returns the next accumulator (either as the raw value or wrapped
        in ``{"accumulator": ...}``).
        """
        accumulator = initial
        for i in range(0, len(items), max(1, chunk_size)):
            chunk = items[i : i + chunk_size]
            response = await self.ctx.call(
                reducer_agent,
                {"accumulator": accumulator, "chunk": chunk},
            )
            if isinstance(response, dict) and "accumulator" in response:
                accumulator = response["accumulator"]
            else:
                accumulator = response
        return accumulator

    async def race(
        self,
        agent_calls: list[tuple[str, dict]],
        *,
        timeout: float = 30.0,
    ) -> Any:
        """Call multiple agents in parallel, return the first successful result.

        All other tasks are cancelled as soon as one succeeds.
        """
        if not agent_calls:
            raise ValueError("race() requires at least one agent call")

        tasks: list[asyncio.Task] = [
            asyncio.create_task(self.ctx.call(name, input_data))
            for name, input_data in agent_calls
        ]
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

        if not done:
            raise asyncio.TimeoutError(
                f"race() — all {len(agent_calls)} agents timed out after {timeout}s"
            )
        # Prefer a successful task; otherwise re-raise the first exception
        first_exc: Exception | None = None
        for task in done:
            exc = task.exception()
            if exc is None:
                return task.result()
            if first_exc is None:
                first_exc = exc
        raise first_exc or RuntimeError("race() — all racers errored")

    async def gather(
        self,
        agent_calls: list[tuple[str, dict]],
        *,
        timeout: float = 60.0,
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Call multiple agents in parallel and wait for all of them.

        Order of results matches the input order. If
        ``return_exceptions`` is True, failed calls produce the exception
        object instead of raising.
        """
        coros = [
            asyncio.wait_for(self.ctx.call(name, input_data), timeout=timeout)
            for name, input_data in agent_calls
        ]
        results = await asyncio.gather(
            *coros, return_exceptions=return_exceptions
        )
        return list(results)
