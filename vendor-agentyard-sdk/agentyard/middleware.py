"""Before/after hooks for agent processing."""

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger("agentyard.middleware")

# Global hook registries
_before_hooks: list[Callable] = []
_after_hooks: list[Callable] = []
_error_hooks: list[Callable] = []


def before(func: Callable) -> Callable:
    """Register a before-processing hook. Receives (input, ctx) and can modify input."""
    _before_hooks.append(func)
    return func


def after(func: Callable) -> Callable:
    """Register an after-processing hook. Receives (input, output, ctx) and can modify output."""
    _after_hooks.append(func)
    return func


def on_error(func: Callable) -> Callable:
    """Register an error hook. Receives (input, error, ctx)."""
    _error_hooks.append(func)
    return func


async def run_before_hooks(input_data: dict, ctx: Any = None) -> dict:
    """Run all before hooks, allow them to modify input."""
    for hook in _before_hooks:
        try:
            if asyncio.iscoroutinefunction(hook):
                result = await hook(input_data, ctx)
            else:
                result = hook(input_data, ctx)
            if isinstance(result, dict):
                input_data = result
        except Exception as exc:
            logger.warning("before_hook_failed hook=%s: %s", hook.__name__, exc)
    return input_data


async def run_after_hooks(
    input_data: dict, output: dict, ctx: Any = None
) -> dict:
    """Run all after hooks, allow them to modify output."""
    for hook in _after_hooks:
        try:
            if asyncio.iscoroutinefunction(hook):
                result = await hook(input_data, output, ctx)
            else:
                result = hook(input_data, output, ctx)
            if isinstance(result, dict):
                output = result
        except Exception as exc:
            logger.warning("after_hook_failed hook=%s: %s", hook.__name__, exc)
    return output


async def run_error_hooks(
    input_data: dict, error: Exception, ctx: Any = None
) -> None:
    """Run all error hooks."""
    for hook in _error_hooks:
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(input_data, error, ctx)
            else:
                hook(input_data, error, ctx)
        except Exception as exc:
            logger.warning("error_hook_failed hook=%s: %s", hook.__name__, exc)
