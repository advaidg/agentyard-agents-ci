"""@yard.agent decorator — collects metadata and builds agent card for publishing.

Schemas can be auto-derived from Pydantic type hints:

    from pydantic import BaseModel

    class InvoiceInput(BaseModel):
        file_url: str
        currency: str = "USD"

    class InvoiceOutput(BaseModel):
        total: float
        line_items: list[dict]

    @yard.agent(name="invoice-parser")
    async def parse(input: InvoiceInput) -> InvoiceOutput:
        ...
    # input_schema and output_schema auto-derived from Pydantic models
"""

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any


def _infer_schemas_from_signature(
    func: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Inspect a function signature and extract JSON schemas from Pydantic models.

    Returns (input_schema, output_schema) — either may be None if not inferable.
    Only the first parameter is considered as input. Return annotation is output.
    Context parameter (YardContext) is skipped.
    """
    try:
        sig = inspect.signature(func)
        hints = typing.get_type_hints(func)
    except Exception:
        return None, None

    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    # First parameter that isn't named 'ctx' or 'context'
    for name, param in sig.parameters.items():
        if name in ("ctx", "context", "self"):
            continue
        annotation = hints.get(name)
        if annotation is not None:
            input_schema = _pydantic_to_schema(annotation)
        break

    # Return annotation
    return_annotation = hints.get("return")
    if return_annotation is not None:
        output_schema = _pydantic_to_schema(return_annotation)

    return input_schema, output_schema


def _pydantic_to_schema(annotation: Any) -> dict[str, Any] | None:
    """Convert a Pydantic model class to a JSON Schema dict. Returns None otherwise."""
    if annotation is None or annotation is type(None):
        return None
    # Check if it's a Pydantic BaseModel subclass
    try:
        from pydantic import BaseModel

        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
    except ImportError:
        pass
    except Exception:
        pass
    return None


@dataclass
class AgentSkill:
    name: str
    description: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


@dataclass
class AgentMetadata:
    name: str
    namespace: str
    description: str
    version: str
    framework: str
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    a2a_endpoint: str = ""
    a2a_url: str = ""
    docker_image: str | None = None
    owner: str = "unknown"
    memory: bool = False
    stateful: bool = False
    skills: list[AgentSkill] = field(default_factory=list)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    # Deployment spec (v2)
    image: str | None = None
    port: int = 9000
    resources: dict[str, str] = field(default_factory=dict)
    replicas: int = 1
    env: dict[str, str] = field(default_factory=dict)
    health_path: str = "/health"
    tools: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    streaming: bool = False
    # Output control
    output_mode: str = "sync"  # sync | async | stream
    output_format: str = "json"  # json | text | binary
    max_output_size: int = 1024 * 1024  # 1MB default
    _func: Any = None

    def to_registration_payload(self) -> dict:
        """Convert to the JSON payload expected by POST /api/agents."""
        # Auto-generate a2a_endpoint from name + port if not set
        if not self.a2a_endpoint:
            self.a2a_endpoint = f"http://{self.name}:{self.port}"
        if not self.a2a_url:
            self.a2a_url = f"{self.a2a_endpoint.rstrip('/')}/.well-known/agent.json"

        # Include event subscriptions in the agent card so other agents
        # know what this agent reacts to.
        try:
            from agentyard.events import get_registered_subscriptions

            subscriptions = [
                {"pattern": s.pattern, "description": s.description}
                for s in get_registered_subscriptions()
            ]
        except Exception:
            subscriptions = []

        agent_card = {
            "name": self.name,
            "description": self.description,
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": s.input_schema,
                    "output_schema": s.output_schema,
                }
                for s in self.skills
            ],
            "subscriptions": subscriptions,
            "auth_schemes": ["bearer"],
            "protocol_version": "0.2",
        }

        return {
            "name": self.name,
            "namespace": self.namespace,
            "description": self.description,
            "version": self.version,
            "framework": self.framework,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "a2a_url": self.a2a_url,
            "a2a_endpoint": self.a2a_endpoint,
            "docker_image": self.docker_image or self.image,
            "agent_card": agent_card,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "owner": self.owner,
            "memory": self.memory,
            "stateful": self.stateful,
            "deployment": {
                "image": self.image or self.docker_image,
                "port": self.port,
                "resources": self.resources,
                "replicas": self.replicas,
                "env": self.env,
                "health_path": self.health_path,
                "tools": self.tools,
                "secrets": self.secrets,
                "streaming": self.streaming,
            },
            "output_config": {
                "mode": self.output_mode,
                "format": self.output_format,
                "max_size": self.max_output_size,
            },
        }


@dataclass
class McpServerMetadata:
    name: str
    description: str = ""
    url: str = ""
    icon: str = "🔧"
    category: str = "utility"
    port: int = 9010
    image: str | None = None
    tools: list[dict] = field(default_factory=list)
    _func: Any = None

    def to_registration_payload(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "icon": self.icon,
            "category": self.category,
        }

    def get_tools(self) -> list[dict]:
        return self.tools


# Global registries (per-process)
_registered_agents: list[AgentMetadata] = []
_registered_mcp_servers: list[McpServerMetadata] = []


def get_registered_agents() -> list[AgentMetadata]:
    """Get all agents registered via @yard.agent in this process."""
    return _registered_agents


def get_registered_mcp_servers() -> list[McpServerMetadata]:
    """Get all MCP servers registered via @yard.mcp_server in this process."""
    return _registered_mcp_servers


class _YardNamespace:
    """Namespace for the @yard.agent decorator and related methods."""

    @staticmethod
    def agent(
        name: str,
        namespace: str = "default",
        description: str = "",
        version: str = "0.1.0",
        framework: str = "custom",
        capabilities: list[str] | None = None,
        tags: list[str] | None = None,
        a2a_endpoint: str = "",
        a2a_url: str = "",
        docker_image: str | None = None,
        owner: str = "unknown",
        memory: bool = False,
        stateful: bool = False,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        # Deployment spec (v2)
        image: str | None = None,
        port: int = 9000,
        resources: dict[str, str] | None = None,
        replicas: int = 1,
        env: dict[str, str] | None = None,
        health_path: str = "/health",
        tools: list[str] | None = None,
        secrets: list[str] | None = None,
        streaming: bool = False,
        # Output control
        output_mode: str = "sync",
        output_format: str = "json",
        max_output_size: int = 1024 * 1024,
    ):
        """Decorator that registers an agent function with AgentYard metadata."""

        def decorator(func):
            # Auto-extract description from docstring if not provided
            desc = description or (inspect.getdoc(func) or f"Agent: {name}")

            # Auto-infer schemas from Pydantic type hints when caller didn't provide any
            inferred_in, inferred_out = _infer_schemas_from_signature(func)
            effective_input_schema = input_schema or inferred_in
            effective_output_schema = output_schema or inferred_out

            # Auto-create a skill from the function
            skills = [
                AgentSkill(
                    name=func.__name__,
                    description=desc,
                    input_schema=effective_input_schema,
                    output_schema=effective_output_schema,
                )
            ]

            metadata = AgentMetadata(
                name=name,
                namespace=namespace,
                description=desc,
                version=version,
                framework=framework,
                capabilities=capabilities or [],
                tags=tags or [],
                a2a_endpoint=a2a_endpoint,
                a2a_url=a2a_url,
                docker_image=docker_image,
                owner=owner,
                memory=memory,
                stateful=stateful,
                skills=skills,
                input_schema=effective_input_schema,
                output_schema=effective_output_schema,
                image=image,
                port=port,
                resources=resources or {},
                replicas=replicas,
                env=env or {},
                health_path=health_path,
                tools=tools or [],
                secrets=secrets or [],
                streaming=streaming,
                output_mode=output_mode,
                output_format=output_format,
                max_output_size=max_output_size,
                _func=func,
            )

            _registered_agents.append(metadata)

            # Attach metadata to the function for introspection
            func._agentyard_metadata = metadata

            return func

        return decorator

    @staticmethod
    def skill(
        name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ):
        """Add an additional skill to the most recently decorated agent."""

        def decorator(func):
            if _registered_agents:
                _registered_agents[-1].skills.append(
                    AgentSkill(
                        name=name,
                        description=description or inspect.getdoc(func) or f"Skill: {name}",
                        input_schema=input_schema,
                        output_schema=output_schema,
                    )
                )
            func._agentyard_skill = True
            return func

        return decorator


    @staticmethod
    def mcp_server(
        name: str,
        description: str = "",
        url: str = "",
        icon: str = "🔧",
        category: str = "utility",
        port: int = 9010,
        image: str | None = None,
    ):
        """Decorator that registers an MCP tools server with AgentYard.

        Usage:
            @yard.mcp_server(name="My Tools", category="utility", port=9010)
            def setup_tools():
                return [
                    {"name": "tool1", "description": "Does X", "category": "utility"},
                    {"name": "tool2", "description": "Does Y", "category": "text"},
                ]
        """
        def decorator(func):
            tools = func()  # Call to get tool definitions
            metadata = McpServerMetadata(
                name=name,
                description=description or inspect.getdoc(func) or f"MCP Server: {name}",
                url=url or f"http://{name.lower().replace(' ', '-')}:{port}/mcp",
                icon=icon,
                category=category,
                port=port,
                image=image,
                tools=tools if isinstance(tools, list) else [],
                _func=func,
            )
            _registered_mcp_servers.append(metadata)
            func._agentyard_mcp_metadata = metadata
            return func
        return decorator

    @staticmethod
    def on(pattern: str, description: str = ""):
        """Subscribe to events matching a Redis channel pattern.

        Usage:
            @yard.on("agent:invoice-processor:complete")
            async def handler(event, ctx):
                ...

            @yard.on("memory:changed:user_profile:*")
            async def on_change(event, ctx):
                ...
        """
        from agentyard.events import on as _on

        return _on(pattern, description)

    @staticmethod
    def subscribe(topic: Any, description: str = ""):
        """Subscribe to a typed topic via a Pydantic-validated message.

        Usage::

            from pydantic import BaseModel
            from agentyard import yard, Topic

            class InvoiceProcessed(BaseModel):
                invoice_id: str
                total: float

            invoices = Topic("invoices.processed", InvoiceProcessed)

            @yard.subscribe(invoices)
            async def on_invoice(msg: InvoiceProcessed, ctx):
                ...
        """
        from agentyard.topics import subscribe as _subscribe

        return _subscribe(topic, description)

    @staticmethod
    def cache(
        ttl: int = 300,
        key: Any = None,
        *,
        namespace: str | None = None,
        on_miss: str | None = None,
    ):
        """Cache an agent handler's output in Redis.

        Usage::

            @yard.cache(ttl=300, key=lambda input, ctx=None: input["url"])
            @yard.agent(name="invoice-parser")
            async def parse(input, ctx):
                ...

        Falls back to a no-op when Redis is not configured so tests
        and local dev keep working without a broker.
        """
        from agentyard.cache import cache as _cache

        return _cache(ttl=ttl, key=key, namespace=namespace, on_miss=on_miss)

    @staticmethod
    def run() -> None:
        """Start the agent or MCP server runtime. Auto-selects transport from YARD_TRANSPORT."""
        from agentyard.runtime import run

        run()


yard = _YardNamespace()
