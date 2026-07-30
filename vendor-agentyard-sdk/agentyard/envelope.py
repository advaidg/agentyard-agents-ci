"""Output envelope — wraps every agent result in a standard metadata envelope.

The envelope carries tracing, timing, and context information alongside
the raw agent output so downstream consumers can inspect provenance
without parsing the payload itself.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class OutputEnvelope:
    """Standard wrapper around an agent's raw output."""

    output: Any
    agent_name: str
    invocation_id: str
    system_id: str
    node_id: str
    duration_ms: int
    timestamp: str  # ISO 8601
    trace_id: str  # For distributed tracing
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the envelope to a plain dictionary.

        The ``output`` key contains the raw result for A2A protocol
        compatibility.  All other keys are envelope metadata.
        """
        return {
            "output": self.output,
            "agent_name": self.agent_name,
            "invocation_id": self.invocation_id,
            "system_id": self.system_id,
            "node_id": self.node_id,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }

    def unwrap(self) -> Any:
        """Return just the raw output, discarding envelope metadata."""
        return self.output


def wrap_output(
    output: Any,
    *,
    agent_name: str,
    invocation_id: str = "",
    system_id: str = "",
    node_id: str = "",
    duration_ms: int = 0,
    trace_id: str = "",
    metadata: dict | None = None,
) -> OutputEnvelope:
    """Create an ``OutputEnvelope`` from context values and a raw result.

    Parameters
    ----------
    output:
        The raw agent result.
    agent_name:
        Name of the agent that produced the output.
    invocation_id:
        Unique identifier for this invocation.
    system_id:
        The AgentYard system that owns this invocation.
    node_id:
        The node within the system graph.
    duration_ms:
        How long the agent took to produce the result.
    trace_id:
        Distributed tracing identifier.  Generated if omitted.
    metadata:
        Arbitrary extra fields to attach.
    """
    return OutputEnvelope(
        output=output,
        agent_name=agent_name,
        invocation_id=invocation_id or str(uuid.uuid4()),
        system_id=system_id,
        node_id=node_id,
        duration_ms=duration_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
        trace_id=trace_id or str(uuid.uuid4()),
        metadata=metadata or {},
    )
