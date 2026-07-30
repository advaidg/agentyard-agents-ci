# AgentYard SDK

Python SDK for building, registering, and managing A2A agents on the AgentYard platform.

## Installation

```bash
# From PyPI (when published)
pip install agentyard

# From local source
pip install ./backend/sdk
```

## Quick Start

```python
from agentyard import yard

@yard.agent(
    name="summarizer",
    namespace="default",
    description="Summarizes text input",
    version="1.0.0",
    framework="custom",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    output_schema={
        "type": "object",
        "properties": {"summary": {"type": "string"}},
    },
)
async def summarize(input: dict) -> dict:
    text = input["text"]
    return {"summary": text[:200] + "..."}

# Start the agent (HTTP server on port 9000 by default)
yard.run()
```

## Features

### Context Object

Agent functions can optionally accept a `YardContext` for access to shared memory, tools, progress streaming, and structured logging:

```python
from agentyard import yard, YardContext

@yard.agent(name="smart-agent", ...)
async def handler(input: dict, ctx: YardContext = None) -> dict:
    # Read/write shared memory (Redis-backed in systems)
    prev = await ctx.memory.get("previous_output") if ctx else None
    if ctx:
        await ctx.memory.set("my_key", {"data": "value"})

    # Use MCP tools (sidecar discovery via YARD_MCP_TOOLS env)
    if ctx and ctx.tools:
        result = await ctx.tools.execute("search_code", {"q": "bug"})

    # Emit streaming progress events
    if ctx:
        await ctx.emit_progress({"status": "halfway", "pct": 50})

    # Structured logging (feeds into AgentYard monitoring)
    if ctx:
        ctx.log("Processing complete", level="info", tokens=150)

    return {"result": "done"}
```

Context is automatically injected by both HTTP and Redis Stream transports when the agent runs inside a system.

### Shared Memory

When agents run as nodes in a system, they share memory via Redis:

```python
# Read a value set by a previous node
value = await ctx.memory.get("analysis_result")

# Write a value for downstream nodes
await ctx.memory.set("my_output", {"score": 0.95})

# Read all shared memory
all_data = await ctx.memory.get_all()

# Delete a key
await ctx.memory.delete("temp_key")
```

Memory strategies (set via `YARD_MEMORY` env var):
- `shared_bus` (default) — all nodes read/write freely
- `isolated` / `none` — writes are silently dropped

### MCP Tools

Agents can call MCP tool servers deployed as sidecars:

```python
from agentyard import ToolsClient

tools = ToolsClient()  # Reads YARD_MCP_TOOLS="github:3100,slack:3101"

# List available tools
all_tools = await tools.list_tools()
github_tools = await tools.list_tools(server="github")

# Execute a tool
result = await tools.execute("create_issue", {"title": "Bug", "body": "..."})
result = await tools.execute("send_message", {"channel": "#dev"}, server="slack")
```

### Input/Output Validation

Schemas declared in `@yard.agent()` are validated automatically on every request:

```python
@yard.agent(
    name="parser",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "max_length": {"type": "integer"},
        },
        "required": ["text"],
    },
    output_schema={
        "type": "object",
        "properties": {"parsed": {"type": "object"}},
    },
)
def parse(input: dict) -> dict:
    ...
```

Invalid input returns HTTP 400 with the validation error message.

### Middleware (Before/After Hooks)

Register hooks that run before and after every invocation:

```python
from agentyard import before, after, on_error

@before
def add_timestamp(input_data, ctx):
    input_data["_received_at"] = "2024-01-01T00:00:00Z"
    return input_data  # Return modified input

@after
def add_metadata(input_data, output, ctx):
    output["_version"] = "1.0"
    return output  # Return modified output

@on_error
def log_failure(input_data, error, ctx):
    print(f"Agent failed: {error}")
```

Hooks support both sync and async functions.

### Metrics

Agent invocations are automatically recorded to Redis for AgentYard analytics:
- Total calls, success/error counts
- Cumulative and per-call latency
- Rolling window of last 1000 latencies

No configuration needed — metrics are collected automatically when `YARD_REDIS_URL` is set.

### Structured Logging

```python
from agentyard import get_logger

log = get_logger("my-agent")
log.info("Processing request", tokens=150, model="gpt-4")
log.warning("Slow response", latency_ms=5000)
log.error("Failed to call downstream", error="timeout")
```

Outputs JSON to stderr, compatible with AgentYard log collection:
```json
{"ts": "2024-01-01T00:00:00Z", "level": "info", "agent": "my-agent", "msg": "Processing request", "tokens": 150}
```

### Testing

Test agents locally without Docker, Redis, or any infrastructure:

```python
from agentyard.testing import test_agent, AgentTestClient

# Quick test
result = test_agent(summarize, {"text": "Hello world"})
assert "summary" in result

# Test client with agent card inspection
client = AgentTestClient(summarize)
result = client.invoke({"text": "Hello"})
card = client.agent_card()
health = client.health()
```

Schema validation runs during tests by default. Disable with `validate=False`:

```python
result = test_agent(handler, {"raw": "data"}, validate=False)
```

### Transport Modes

Set `YARD_TRANSPORT` to choose how the agent receives traffic:

| Value | Description |
|-------|-------------|
| `http` (default) | FastAPI server with A2A endpoints |
| `redis-stream` | Redis Stream consumer (requires `YARD_SYSTEM_ID` + `YARD_NODE_ID`) |
| `both` | HTTP for health checks + Redis for production traffic |

## CLI Commands

### `agentyard publish`

Register agents with the AgentYard registry:

```bash
agentyard publish -f my_agent.py
agentyard publish -m my_package.agent
```

### `agentyard build`

Build a Docker image for an agent:

```bash
agentyard build -f my_agent.py
agentyard build -f my_agent.py -t myrepo/agent:1.0
agentyard build -f my_agent.py --push
```

### `agentyard list`

```bash
agentyard list
agentyard list --namespace acme --framework langchain
agentyard list -q "invoice parser" --limit 10
```

### `agentyard info`

```bash
agentyard info invoice-parser
agentyard info 550e8400-e29b-41d4-a716-446655440000
```

### `agentyard health`

```bash
agentyard health invoice-parser
```

### `agentyard deprecate`

```bash
agentyard deprecate 550e8400... --note "Replaced by v2"
```

### `agentyard stats`

```bash
agentyard stats
```

### `agentyard config`

```bash
agentyard config set registry-url http://localhost:8000
agentyard config set token ayard_tok_abc123
agentyard config get registry-url
agentyard config show
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YARD_TRANSPORT` | `http` | Transport mode: `http`, `redis-stream`, `both` |
| `YARD_PORT` | `9000` | HTTP server port |
| `YARD_REDIS_URL` | `redis://redis:6379` | Redis connection URL |
| `YARD_SYSTEM_ID` | | System ID (required for redis-stream) |
| `YARD_NODE_ID` | | Node ID within system (required for redis-stream) |
| `YARD_MEMORY` | `shared_bus` | Memory strategy: `shared_bus`, `isolated`, `none` |
| `YARD_MCP_TOOLS` | | MCP sidecar discovery: `github:3100,slack:3101` |
| `YARD_AGENT_NAME` | | Agent name for logging |
| `AGENTYARD_REGISTRY_URL` | `http://registry:8001` | Registry URL for auto-registration |
| `AGENTYARD_URL` | | Alternative registry URL |

## Architecture

```
@yard.agent decorator
    |
    v
yard.run() --> selects transport
    |
    +-- http_adapter.py --> FastAPI server
    |       - /.well-known/agent.json (A2A agent card)
    |       - POST / (process input)
    |       - GET /health
    |
    +-- redis_adapter.py --> Redis Stream consumer
            - Reads from yard:system:{id}:node:{id}:in
            - Writes to yard:system:{id}:node:{id}:out
            - Traces to yard:system:{id}:trace

Both adapters:
    - Create YardContext with memory, tools, logging
    - Run before/after middleware hooks
    - Validate input/output schemas
    - Record metrics to Redis
```
