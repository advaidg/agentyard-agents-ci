"""Saga pattern for long-running transactional workflows.

A saga is a sequence of steps where each step has:
- forward: the action to perform
- compensate: how to undo it on failure (optional)

On any step failure, compensations run in reverse for all completed steps.
The pattern is the standard answer to "how do I do transactions across
agent calls when each call has a side effect"?

Usage:
    from agentyard import yard, Saga

    @yard.agent(name="order-pipeline")
    async def place_order(input, ctx):
        saga = Saga(ctx, name="place_order")

        saga.step(
            name="reserve_inventory",
            forward=lambda: ctx.call("inventory", {"reserve": input["items"]}),
            compensate=lambda result: ctx.call(
                "inventory", {"release": result["reservation_id"]}
            ),
        )
        saga.step(
            name="charge_card",
            forward=lambda: ctx.call("payments", {"charge": input["amount"]}),
            compensate=lambda result: ctx.call(
                "payments", {"refund": result["charge_id"]}
            ),
            retries=2,
        )
        saga.step(
            name="send_confirmation",
            forward=lambda: ctx.call("email", {"to": input["email"]}),
            # No compensate — email isn't reversible
        )

        return await saga.execute()
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger("agentyard.saga")


@dataclass
class SagaStep:
    name: str
    forward: Callable[[], Awaitable[Any]]
    compensate: Callable[[Any], Awaitable[Any]] | None = None
    retries: int = 0
    result: Any = None
    status: str = "pending"  # pending | completed | failed | compensated | skipped
    error: str | None = None


class SagaCompensationError(Exception):
    """Raised when a compensation handler itself fails."""


class SagaAbortError(Exception):
    """Raised when a saga aborts after compensations have run."""

    def __init__(self, step_name: str, cause: Exception, compensations_ok: bool):
        self.step_name = step_name
        self.cause = cause
        self.compensations_ok = compensations_ok
        super().__init__(
            f"Saga aborted at step '{step_name}': {cause} "
            f"(compensations_ok={compensations_ok})"
        )


class Saga:
    """A multi-step compensation-aware workflow."""

    def __init__(self, ctx: Any = None, name: str = "saga"):
        self.ctx = ctx
        self.name = name
        self.steps: list[SagaStep] = []

    def step(
        self,
        name: str,
        forward: Callable[[], Awaitable[Any]],
        compensate: Callable[[Any], Awaitable[Any]] | None = None,
        retries: int = 0,
    ) -> "Saga":
        """Append a step. Returns self for chaining."""
        self.steps.append(
            SagaStep(name=name, forward=forward, compensate=compensate, retries=retries)
        )
        return self

    async def execute(self) -> dict:
        """Run every step in order. Compensate on first failure.

        Returns a summary dict with per-step status. Raises SagaAbortError
        if any step ultimately fails after retries.
        """
        completed: list[SagaStep] = []
        for step in self.steps:
            logger.info("saga_step_start saga=%s step=%s", self.name, step.name)
            success = False
            last_exc: Exception | None = None
            for attempt in range(step.retries + 1):
                try:
                    step.result = await step.forward()
                    step.status = "completed"
                    completed.append(step)
                    success = True
                    logger.info(
                        "saga_step_done saga=%s step=%s attempt=%d",
                        self.name,
                        step.name,
                        attempt,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < step.retries:
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
            if not success:
                step.status = "failed"
                step.error = str(last_exc) if last_exc else "unknown"
                logger.error(
                    "saga_step_failed saga=%s step=%s err=%s",
                    self.name,
                    step.name,
                    step.error,
                )
                # Mark remaining steps as skipped
                for remaining in self.steps[len(completed) + 1 :]:
                    remaining.status = "skipped"
                comp_ok = await self._compensate_all(completed)
                raise SagaAbortError(
                    step.name, last_exc or Exception("unknown"), comp_ok
                )
        return self._summary()

    async def _compensate_all(self, completed: list[SagaStep]) -> bool:
        """Run compensations in reverse. Returns True iff all succeeded."""
        all_ok = True
        for step in reversed(completed):
            if step.compensate is None:
                continue
            try:
                logger.info(
                    "saga_compensate saga=%s step=%s", self.name, step.name
                )
                await step.compensate(step.result)
                step.status = "compensated"
            except Exception as exc:
                all_ok = False
                logger.error(
                    "saga_compensate_failed saga=%s step=%s err=%s",
                    self.name,
                    step.name,
                    exc,
                )
        return all_ok

    def _summary(self) -> dict:
        return {
            "name": self.name,
            "status": "completed",
            "steps": [
                {
                    "name": s.name,
                    "status": s.status,
                    "result": s.result,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }
