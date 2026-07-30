"""Local testing utilities — test agents without Docker or Redis.

Usage:
    from agentyard.testing import test_agent

    result = test_agent(my_handler, {"text": "hello world"})
    assert result["summary"] is not None
"""

import asyncio
import inspect
from typing import Any

from agentyard.context import YardContext
from agentyard.validation import validate_input, validate_output


def test_agent(handler: Any, input_data: dict, validate: bool = True) -> dict:
    """Test an agent handler locally without any infrastructure.

    Args:
        handler: The @yard.agent decorated function
        input_data: Test input
        validate: Whether to validate against declared schemas

    Returns:
        Agent output dict
    """
    metadata = getattr(handler, "_agentyard_metadata", None)

    # Validate input schema
    if validate and metadata and metadata.input_schema:
        valid, error = validate_input(input_data, metadata.input_schema)
        if not valid:
            raise ValueError(f"Input validation failed: {error}")

    # Create a mock context
    ctx = YardContext(
        invocation_id="test-invocation",
        system_id="test-system",
        node_id="test-node",
        agent_name=metadata.name if metadata else "test",
    )

    # Call the handler
    if inspect.iscoroutinefunction(handler):
        result = asyncio.run(_call_async(handler, input_data, ctx))
    else:
        sig = inspect.signature(handler)
        params = list(sig.parameters.keys())
        if len(params) > 1:
            result = handler(input_data, ctx)
        else:
            result = handler(input_data)

    # Validate output schema
    if validate and metadata and metadata.output_schema and isinstance(result, dict):
        valid, error = validate_output(result, metadata.output_schema)
        if not valid:
            raise ValueError(f"Output validation failed: {error}")

    return result


async def _call_async(handler: Any, input_data: dict, ctx: YardContext) -> Any:
    sig = inspect.signature(handler)
    params = list(sig.parameters.keys())
    if len(params) > 1:
        return await handler(input_data, ctx)
    return await handler(input_data)


class AgentTestClient:
    """Test client for A2A HTTP mode."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.metadata = getattr(handler, "_agentyard_metadata", None)

    def invoke(self, input_data: dict) -> dict:
        return test_agent(self.handler, input_data)

    def agent_card(self) -> dict:
        if self.metadata:
            return self.metadata.to_registration_payload()["agent_card"]
        return {}

    def health(self) -> dict:
        name = self.metadata.name if self.metadata else "test"
        return {"status": "ok", "agent": name, "transport": "test"}
