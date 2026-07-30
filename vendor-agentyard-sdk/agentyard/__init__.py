"""AgentYard SDK — register, discover, and manage A2A agents."""

from agentyard.cache import cache as yard_cache
from agentyard.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from agentyard.client_a2a import A2AClient, A2ACallError
from agentyard.context import (
    CheckpointRejectedError,
    CheckpointTimeoutError,
    MemoryAccessError,
    MemoryClient,
    YardContext,
)
from agentyard.decorator import yard
from agentyard.emit import Emitter, EmitTarget
from agentyard.lock import DistributedLock, LockAcquireError, LockLostError
from agentyard.envelope import OutputEnvelope, wrap_output
from agentyard.events import EventSubscriber, EventSubscription
from agentyard.llm import (
    LLMChunk,
    LLMClient,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
)
from agentyard.llm import Message as LLMMessage
from agentyard.logging import get_logger
from agentyard.metrics import MetricsReporter
from agentyard.middleware import after, before, on_error
from agentyard.parallel import MapResult, ParallelPrimitives
from agentyard.prompts import PromptClientError, PromptsClient, RenderedPrompt
from agentyard.reasoning import (
    Reasoner,
    ReasoningError,
    ReasoningResult,
    ReasoningStep,
)
from agentyard.register import auto_register
from agentyard.saga import Saga, SagaAbortError, SagaCompensationError, SagaStep
from agentyard.scheduling import (
    ScheduleError,
    Scheduler,
    WaitTimeoutError,
)
from agentyard.testing import AgentTestClient, test_agent
from agentyard.tools import ToolExecutionError, ToolNotFoundError, ToolsClient
from agentyard.topics import (
    Topic,
    TopicMessage,
    TopicPublisher,
    TopicSubscriber,
    TopicSubscription,
)
from agentyard.tracing import Span, Tracer
from agentyard.vector_store import (
    VectorHit,
    VectorItem,
    VectorStoreClient,
    VectorStoreError,
)

__all__ = [
    "yard",
    "auto_register",
    "Emitter",
    "EmitTarget",
    "EventSubscriber",
    "EventSubscription",
    "OutputEnvelope",
    "wrap_output",
    "ToolsClient",
    "ToolNotFoundError",
    "ToolExecutionError",
    "YardContext",
    "MemoryClient",
    "MemoryAccessError",
    "CheckpointRejectedError",
    "CheckpointTimeoutError",
    "test_agent",
    "AgentTestClient",
    "get_logger",
    "before",
    "after",
    "on_error",
    "MetricsReporter",
    "Tracer",
    "Span",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "A2AClient",
    "A2ACallError",
    "PromptsClient",
    "RenderedPrompt",
    "PromptClientError",
    "LLMClient",
    "LLMResponse",
    "LLMChunk",
    "LLMMessage",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "VectorStoreClient",
    "VectorHit",
    "VectorItem",
    "VectorStoreError",
    "DistributedLock",
    "LockAcquireError",
    "LockLostError",
    "yard_cache",
    "Topic",
    "TopicMessage",
    "TopicPublisher",
    "TopicSubscriber",
    "TopicSubscription",
    "Saga",
    "SagaStep",
    "SagaAbortError",
    "SagaCompensationError",
    "Scheduler",
    "ScheduleError",
    "WaitTimeoutError",
    "ParallelPrimitives",
    "MapResult",
    "Reasoner",
    "ReasoningResult",
    "ReasoningStep",
    "ReasoningError",
]
__version__ = "0.6.0"
