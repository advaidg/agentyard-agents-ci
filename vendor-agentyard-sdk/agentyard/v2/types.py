"""Type definitions for AgentYard v2 capability declarations."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceKind(str, Enum):
    """Standard resource types agents can declare needing."""
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    S3 = "s3"
    LLM = "llm"
    SECRETS = "secrets"
    FILESYSTEM = "filesystem"
    GPU = "gpu"
    HTTP_OUTBOUND = "http_outbound"


@dataclass(frozen=True)
class Resource:
    """A resource the agent needs (DB, LLM, secrets, etc.)."""
    kind: ResourceKind
    name: str = ""  # binding name (e.g. "primary_db")
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def postgres(cls, name: str = "default", **opts) -> "Resource":
        return cls(ResourceKind.POSTGRES, name, opts)

    @classmethod
    def redis(cls, name: str = "default", **opts) -> "Resource":
        return cls(ResourceKind.REDIS, name, opts)

    @classmethod
    def llm(cls, provider: str = "anthropic", **opts) -> "Resource":
        return cls(ResourceKind.LLM, provider, opts)

    @classmethod
    def secrets(cls, keys: list[str]) -> "Resource":
        return cls(ResourceKind.SECRETS, "default", {"keys": keys})

    @classmethod
    def s3(cls, bucket: str, **opts) -> "Resource":
        return cls(ResourceKind.S3, bucket, opts)


@dataclass(frozen=True)
class MemoryContract:
    """Declares which memory keys this agent reads/writes."""
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    scope: str = "system"  # "system" | "namespace" | "global"


class FailureMode(str, Enum):
    """How the orchestrator handles failures from this agent."""
    ABORT = "abort"
    SKIP = "skip"
    FALLBACK = "fallback"
    RETRY = "retry"
    DLQ = "dlq"
    COMPENSATE = "compensate"


@dataclass(frozen=True)
class FailurePolicy:
    """Failure handling for this agent."""
    mode: FailureMode = FailureMode.RETRY
    max_retries: int = 2
    retry_delay_ms: int = 1000
    fallback_agent: str = ""
    dlq_topic: str = ""
