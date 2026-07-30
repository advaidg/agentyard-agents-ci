"""AgentYard CLI — register, discover, and manage A2A agents from the terminal.

Usage:
    agentyard publish                    # Register agent from @yard.agent decorator
    agentyard list --namespace acme      # List agents
    agentyard info <name-or-id>          # Get agent details
    agentyard health <name-or-id>        # Health check
    agentyard deprecate <id> --note "Use v2"
    agentyard build -f agent.py          # Build Docker image
    agentyard config set registry-url http://localhost:8000
    agentyard config set token ayard_tok_...
    agentyard config get registry-url
    agentyard stats                      # Platform statistics
"""

import importlib
import os
import sys

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentyard.client import AgentYardClient, AgentYardError
from agentyard.config import get_value, load_config, set_value

console = Console()


def _get_client() -> AgentYardClient:
    return AgentYardClient(
        registry_url=get_value("registry_url"),
        token=get_value("token"),
    )


@click.group()
@click.version_option(version="0.3.0", prog_name="agentyard")
def cli():
    """AgentYard CLI — register, discover, and manage A2A agents."""
    pass


# ── publish ──


@cli.command()
@click.option(
    "--module",
    "-m",
    help="Python module containing @yard.agent decorated functions",
)
@click.option(
    "--file",
    "-f",
    "filepath",
    help="Python file containing @yard.agent decorated functions",
)
def publish(module: str | None, filepath: str | None):
    """Publish agents decorated with @yard.agent to the registry."""
    from agentyard.decorator import get_registered_agents

    # Import the module to trigger decorators
    if filepath:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_agentyard_target", filepath)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    elif module:
        importlib.import_module(module)

    agents = get_registered_agents()
    if not agents:
        console.print(
            "[yellow]No agents found.[/yellow] Decorate functions with @yard.agent first."
        )
        return

    client = _get_client()
    for agent in agents:
        payload = agent.to_registration_payload()
        try:
            result = client.register_agent(payload)
            console.print(
                f"[green]Published[/green] {agent.name}@{agent.version} "
                f"-> {result['id'][:8]}..."
            )
        except (AgentYardError, Exception) as e:
            console.print(f"[red]Failed[/red] {agent.name}: {e}")


# ── list ──


@cli.command("list")
@click.option("--namespace", "-n", help="Filter by namespace")
@click.option("--framework", "-f", help="Filter by framework")
@click.option("--query", "-q", help="Full-text search")
@click.option("--limit", "-l", default=20, help="Max results")
def list_agents(
    namespace: str | None,
    framework: str | None,
    query: str | None,
    limit: int,
):
    """List registered agents."""
    client = _get_client()
    try:
        data = client.list_agents(
            namespace=namespace, framework=framework, q=query, limit=limit
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    items = data.get("items", [])
    total = data.get("total", 0)

    table = Table(title=f"Agents ({total} total)", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Namespace", style="dim")
    table.add_column("Version")
    table.add_column("Framework", style="magenta")
    table.add_column("Status")
    table.add_column("Capabilities")

    for agent in items:
        status_style = "green" if agent["status"] == "active" else "red"
        table.add_row(
            agent["name"],
            agent["namespace"],
            agent["version"],
            agent["framework"],
            f"[{status_style}]{agent['status']}[/{status_style}]",
            ", ".join(agent.get("capabilities", [])[:3]),
        )

    console.print(table)


# ── info ──


@cli.command()
@click.argument("agent_id")
def info(agent_id: str):
    """Get detailed info about an agent."""
    client = _get_client()

    # Try as ID first, then search by name
    try:
        agent = client.get_agent(agent_id)
    except Exception:
        try:
            results = client.search_agents(agent_id)
            items = results.get("items", [])
            if not items:
                console.print(f"[red]Agent not found:[/red] {agent_id}")
                return
            agent = items[0]
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return

    # Display agent info
    status_color = "green" if agent["status"] == "active" else "red"
    panel_content = f"""[bold]{agent['name']}[/bold] v{agent['version']}
[dim]{agent['namespace']}[/dim]

{agent['description']}

[bold]Framework:[/bold] {agent['framework']}
[bold]Owner:[/bold] {agent['owner']}
[bold]Status:[/bold] [{status_color}]{agent['status']}[/{status_color}]
[bold]Capabilities:[/bold] {', '.join(agent.get('capabilities', []))}
[bold]Tags:[/bold] {', '.join(agent.get('tags', []))}

[bold]A2A Endpoint:[/bold] {agent.get('a2a_endpoint', 'N/A')}
[bold]Docker Image:[/bold] {agent.get('docker_image', 'N/A')}
[bold]Memory:[/bold] {agent.get('memory', False)} | [bold]Stateful:[/bold] {agent.get('stateful', False)}"""

    console.print(Panel(panel_content, title="Agent Details", border_style="cyan"))

    # Skills
    card = agent.get("agent_card", {})
    skills = card.get("skills", [])
    if skills:
        skill_table = Table(title="Skills", box=box.SIMPLE)
        skill_table.add_column("Name", style="cyan")
        skill_table.add_column("Description")
        for skill in skills:
            skill_table.add_row(skill["name"], skill.get("description", ""))
        console.print(skill_table)


# ── health ──


@cli.command()
@click.argument("agent_id")
def health(agent_id: str):
    """Check an agent's A2A endpoint health."""
    client = _get_client()

    # Resolve name to ID if needed
    try:
        result = client.check_health(agent_id)
    except Exception:
        try:
            results = client.search_agents(agent_id)
            items = results.get("items", [])
            if not items:
                console.print(f"[red]Agent not found:[/red] {agent_id}")
                return
            result = client.check_health(items[0]["id"])
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return

    status = result.get("status", "unknown")
    latency = result.get("latency_ms", "?")
    color = {"healthy": "green", "slow": "yellow", "unreachable": "red"}.get(
        status, "dim"
    )

    console.print(
        f"[{color}]{result.get('agent_name', agent_id)}[/{color}]: "
        f"{status} ({latency}ms)"
    )


# ── deprecate ──


@cli.command()
@click.argument("agent_id")
@click.option("--note", "-n", required=True, help="Deprecation note")
def deprecate(agent_id: str, note: str):
    """Deprecate an agent (soft-delete)."""
    client = _get_client()
    try:
        result = client.deprecate_agent(agent_id, note)
        console.print(
            f"[yellow]Deprecated[/yellow] {result.get('name', agent_id)}: {note}"
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# ── build ──


@cli.command()
@click.option(
    "--file",
    "-f",
    "filepath",
    required=True,
    help="Python file with @yard.agent",
)
@click.option(
    "--tag",
    "-t",
    help="Docker image tag (default: agentyard-{name}:latest)",
)
@click.option("--push", is_flag=True, help="Push to registry after build")
def build(filepath: str, tag: str | None, push: bool):
    """Build a Docker image for an agent."""
    import importlib.util
    import shutil
    import subprocess
    import tempfile

    from agentyard.decorator import get_registered_agents

    # Load the module to get metadata
    spec = importlib.util.spec_from_file_location("_agent", filepath)
    if not spec or not spec.loader:
        console.print(f"[red]Cannot load file:[/red] {filepath}")
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    agents = get_registered_agents()
    if not agents:
        console.print("[red]No @yard.agent found in file[/red]")
        return

    agent = agents[0]
    image_tag = tag or agent.image or f"agentyard-{agent.name}:latest"

    # Generate Dockerfile
    dockerfile_content = f"""FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir agentyard
COPY {os.path.basename(filepath)} main.py
ENV YARD_PORT={agent.port}
CMD ["python", "main.py"]
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy agent file
        shutil.copy(filepath, os.path.join(tmpdir, os.path.basename(filepath)))

        # Write Dockerfile
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)

        # Build
        console.print(f"[cyan]Building[/cyan] {image_tag}")
        result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            console.print(f"[red]Build failed:[/red] {result.stderr}")
            return

        console.print(f"[green]Built[/green] {image_tag}")

        if push:
            console.print(f"[cyan]Pushing[/cyan] {image_tag}")
            result = subprocess.run(
                ["docker", "push", image_tag],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                console.print(f"[green]Pushed[/green] {image_tag}")
            else:
                console.print(f"[red]Push failed:[/red] {result.stderr}")


# ── stats ──


@cli.command()
def stats():
    """Show platform statistics."""
    client = _get_client()
    try:
        data = client.platform_stats()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    table = Table(title="Platform Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="bold")

    for key, value in data.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


# ── config ──


@cli.group()
def config():
    """Manage AgentYard CLI configuration."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value."""
    key_normalized = key.replace("-", "_")
    set_value(key_normalized, value)
    console.print(f"[green]Set[/green] {key} = {value}")


@config.command("get")
@click.argument("key")
def config_get(key: str):
    """Get a configuration value."""
    key_normalized = key.replace("-", "_")
    value = get_value(key_normalized)
    console.print(f"{key} = {value}")


@config.command("show")
def config_show():
    """Show all configuration."""
    cfg = load_config()
    for key, value in cfg.items():
        display_value = (
            value
            if key != "token" or not value
            else (
                f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            )
        )
        console.print(f"  {key}: {display_value}")


# ── init ──
#
# Scaffolds a new agent project directory from in-line string templates.
# We keep templates inline (no separate templates/ dir) so the SDK install
# has no extra package data to ship.


# Immutable template set — rendered with str.format(**context).
# Double-braces `{{` `}}` are literal braces in the output, used when the
# template contains Python f-string or dict syntax that should not be
# format-interpolated.
_MAIN_PY_TEMPLATE = '''"""{name} — AgentYard agent entrypoint.

Generated by `agentyard init`. Edit the handler below, then run:

    agentyard dev              # hot-reload development
    agentyard publish -f main.py  # register with the platform
    agentyard build -f main.py    # build a Docker image
"""

from agentyard import yard


@yard.agent(
    name="{name}",
    namespace="{namespace}",
    description="{description}",
    version="0.1.0",
    framework="{framework}",
    capabilities=["example"],
    port={port},
    # SDK v0.4.0 features you can opt into:
    #   memory=True,            # enable shared memory bus participation
    #   stateful=True,          # keep per-session context between calls
    #   streaming=True,         # emit partial results via ctx.emit()
    #   output_mode="stream",   # sync | async | stream
    #   tools=["search", "http"],  # MCP tools attached at runtime
)
async def handle(input: dict, ctx=None) -> dict:
    """Main agent handler.

    The `ctx` argument is optional. When present it exposes:
        ctx.emit(event)            — stream a partial result (SDK v0.4.0)
        ctx.trace(span_name)       — open a tracing span
        ctx.memory.get(key)        — read from shared memory bus
        ctx.memory.put(key, val)   — write to shared memory bus
    """
    # Example: echo the input back with a greeting.
    message = input.get("message", "world")
    return {{"greeting": f"hello, {{message}}", "agent": "{name}"}}


if __name__ == "__main__":
    # yard.run() auto-selects transport from YARD_TRANSPORT (http | redis-stream | both)
    yard.run()
'''


_PYPROJECT_TEMPLATE = '''[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.10"
dependencies = [
    "agentyard>=0.4.0",
]

[project.optional-dependencies]
dev = [
    "agentyard[dev]>=0.4.0",
]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"
'''


_DOCKERFILE_TEMPLATE = '''FROM python:3.12-slim

WORKDIR /app

# Install the SDK — pin to a known-good version in production.
RUN pip install --no-cache-dir "agentyard>=0.4.0"

COPY main.py .

ENV YARD_PORT={port}
ENV YARD_TRANSPORT=http

EXPOSE {port}

CMD ["python", "main.py"]
'''


_ENV_EXAMPLE_TEMPLATE = '''# AgentYard runtime configuration
# Copy to `.env` and edit as needed.

# Registry the agent auto-registers against on startup.
AGENTYARD_REGISTRY_URL=http://localhost:8080
AGENTYARD_TOKEN=

# Transport: http | redis-stream | both
YARD_TRANSPORT=http

# HTTP port the A2A endpoint listens on.
YARD_PORT={port}

# Output mode: sync | async | stream
YARD_OUTPUT_MODE=sync

# Optional — Redis for stream transport, memory bus, and heartbeats.
REDIS_URL=redis://localhost:6379/0

# Optional — OpenTelemetry tracing.
# OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
# OTEL_SERVICE_NAME={name}
'''


_README_TEMPLATE = '''# {name}

{description}

## Quickstart

```bash
# 1. Install the SDK
pip install -e ".[dev]"

# 2. Copy env template
cp .env.example .env

# 3. Run with hot-reload
agentyard dev

# 4. In another terminal, call the agent
curl -X POST http://localhost:{port}/invoke \\
     -H "Content-Type: application/json" \\
     -d '{{"message": "there"}}'
```

## Publish to the registry

```bash
agentyard publish -f main.py
```

## Build a Docker image

```bash
agentyard build -f main.py -t {name}:0.1.0
```

## Layout

- `main.py` — agent handler decorated with `@yard.agent`
- `Dockerfile` — production container
- `.env.example` — runtime environment variables
- `pyproject.toml` — Python package metadata
'''


_DOCKERIGNORE_TEMPLATE = '''__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.venv/
venv/
env/
.env
.env.*
!.env.example
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/
*.egg-info/
.git/
.gitignore
README.md
'''


_GITIGNORE_TEMPLATE = '''__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
env/
.env
.env.*
!.env.example
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/
*.egg-info/
.DS_Store
.idea/
.vscode/
'''


_INIT_FILES: tuple[tuple[str, str], ...] = (
    ("main.py", _MAIN_PY_TEMPLATE),
    ("pyproject.toml", _PYPROJECT_TEMPLATE),
    ("Dockerfile", _DOCKERFILE_TEMPLATE),
    (".env.example", _ENV_EXAMPLE_TEMPLATE),
    ("README.md", _README_TEMPLATE),
    (".dockerignore", _DOCKERIGNORE_TEMPLATE),
    (".gitignore", _GITIGNORE_TEMPLATE),
)


@cli.command()
@click.argument("name")
@click.option(
    "--framework",
    default="custom",
    show_default=True,
    help="Agent framework (langchain, crewai, custom, ...)",
)
@click.option(
    "--namespace",
    default="default",
    show_default=True,
    help="Agent namespace, e.g. acme/finance",
)
@click.option(
    "--port",
    default=9000,
    show_default=True,
    type=int,
    help="HTTP port the agent listens on",
)
@click.option(
    "--description",
    default="",
    help="Short agent description",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files in the target directory",
)
def init(
    name: str,
    framework: str,
    namespace: str,
    port: int,
    description: str,
    force: bool,
):
    """Scaffold a new AgentYard agent project in ./<NAME>/."""
    target_dir = os.path.abspath(name)
    if os.path.exists(target_dir) and not force:
        # Only fail if the directory has content — empty dirs are fine.
        if any(os.scandir(target_dir)):
            console.print(
                f"[red]Directory not empty:[/red] {target_dir}\n"
                f"Use [bold]--force[/bold] to overwrite."
            )
            sys.exit(1)

    os.makedirs(target_dir, exist_ok=True)

    # Render context used by every template.
    context = {
        "name": name,
        "namespace": namespace,
        "framework": framework,
        "port": port,
        "description": description or f"AgentYard agent: {name}",
    }

    created: list[str] = []
    skipped: list[str] = []
    for filename, template in _INIT_FILES:
        dest = os.path.join(target_dir, filename)
        if os.path.exists(dest) and not force:
            skipped.append(filename)
            continue
        with open(dest, "w", encoding="utf-8") as f:
            f.write(template.format(**context))
        created.append(filename)

    console.print(
        Panel(
            f"[bold green]Created agent[/bold green] [cyan]{name}[/cyan]\n"
            f"[dim]{target_dir}[/dim]\n\n"
            f"Framework: [magenta]{framework}[/magenta]\n"
            f"Namespace: [magenta]{namespace}[/magenta]\n"
            f"Port:      [magenta]{port}[/magenta]\n\n"
            f"Files: {', '.join(created) if created else '(none)'}"
            + (f"\nSkipped: {', '.join(skipped)}" if skipped else ""),
            title="agentyard init",
            border_style="green",
        )
    )
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        f"  [cyan]cd {name}[/cyan]\n"
        "  [cyan]cp .env.example .env[/cyan]\n"
        "  [cyan]pip install -e '.[dev]'[/cyan]\n"
        "  [cyan]agentyard dev[/cyan]"
    )


# ── dev ──
#
# Run the agent with hot-reload. Because `yard.run()` owns its own event loop
# (uvicorn / asyncio), we can't use uvicorn --reload directly. Instead we
# spawn the agent as a subprocess and use `watchfiles` to re-exec it on .py
# changes.


def _load_dotenv_file(path: str) -> dict[str, str]:
    """Minimal .env parser — no external dependency on python-dotenv."""
    env: dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _detect_agent_port(file_path: str, fallback: int) -> int:
    """Inspect the decorated agent to pull its configured port, if any."""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_agentyard_dev_probe", file_path)
        if not spec or not spec.loader:
            return fallback
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from agentyard.decorator import get_registered_agents

        agents = get_registered_agents()
        if agents:
            return agents[0].port
    except Exception:
        pass
    return fallback


@cli.command()
@click.option(
    "--file",
    "-f",
    "filepath",
    default="main.py",
    show_default=True,
    help="Agent entrypoint file",
)
@click.option(
    "--registry",
    default=None,
    help="Registry URL (overrides AGENTYARD_REGISTRY_URL)",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Override YARD_PORT (default: value from decorator or 9000)",
)
@click.option(
    "--no-reload",
    is_flag=True,
    help="Disable file-watching / hot-reload",
)
def dev(
    filepath: str,
    registry: str | None,
    port: int | None,
    no_reload: bool,
):
    """Run an agent locally with hot-reload on .py changes."""
    import signal
    import subprocess

    if not os.path.exists(filepath):
        console.print(f"[red]File not found:[/red] {filepath}")
        sys.exit(1)

    abs_file = os.path.abspath(filepath)
    work_dir = os.path.dirname(abs_file)

    # Resolve port: CLI flag > decorator metadata > 9000.
    resolved_port = port or _detect_agent_port(abs_file, fallback=9000)

    # Build the subprocess environment. Start from current env so user
    # overrides still apply, then layer .env file, then CLI args.
    child_env = os.environ.copy()
    child_env.update(_load_dotenv_file(os.path.join(work_dir, ".env")))
    child_env["YARD_PORT"] = str(resolved_port)
    child_env["YARD_TRANSPORT"] = child_env.get("YARD_TRANSPORT", "http")
    if registry:
        child_env["AGENTYARD_REGISTRY_URL"] = registry
    child_env.setdefault("AGENTYARD_REGISTRY_URL", "http://localhost:8080")

    endpoint = f"http://localhost:{resolved_port}"
    console.print(
        Panel(
            f"[bold cyan]agentyard dev[/bold cyan]\n\n"
            f"Entrypoint:    [white]{abs_file}[/white]\n"
            f"Registry:      [magenta]{child_env['AGENTYARD_REGISTRY_URL']}[/magenta]\n"
            f"A2A endpoint:  [green]{endpoint}[/green]\n"
            f"Agent card:    [green]{endpoint}/.well-known/agent.json[/green]\n"
            f"Hot reload:    {'[dim]off[/dim]' if no_reload else '[green]on[/green]'}",
            title="Dev Server",
            border_style="cyan",
        )
    )

    def _spawn() -> subprocess.Popen:
        console.print("[dim]» starting agent…[/dim]")
        return subprocess.Popen(
            [sys.executable, abs_file],
            cwd=work_dir,
            env=child_env,
        )

    proc = _spawn()

    if no_reload:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
        return

    try:
        from watchfiles import watch
    except ImportError:
        console.print(
            "[yellow]watchfiles not installed[/yellow] — "
            "install with [cyan]pip install 'agentyard[dev]'[/cyan] "
            "for hot-reload. Running without reload."
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
        return

    try:
        for changes in watch(
            work_dir,
            watch_filter=lambda _change, path: path.endswith(".py"),
            stop_event=None,
        ):
            files = ", ".join(sorted({os.path.basename(p) for _, p in changes})[:3])
            console.print(f"[yellow]» change detected[/yellow] ({files}) — restarting")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc = _spawn()
    except KeyboardInterrupt:
        console.print("\n[dim]» stopping agent…[/dim]")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ── scenarios (G10) ──
#
# Scenarios are multi-step test flows attached to a system. Each scenario
# has steps with assertions; `run-all --fail-on-error` is the CI integration
# hook — it exits non-zero if any scenario fails, so teams can wire it into
# GitHub Actions / GitLab CI / etc.


def _resolve_system(client: AgentYardClient, identifier: str) -> dict | None:
    """Resolve a `--system` CLI arg to a system dict (slug, name, or UUID)."""
    try:
        return client.find_system_by_slug_or_name(identifier)
    except Exception as exc:  # noqa: BLE001 — surface the message
        console.print(f"[red]Error resolving system:[/red] {exc}")
        return None


def _scenario_status_style(status: str) -> str:
    return {
        "passed": "green",
        "failed": "red",
        "error": "yellow",
        "running": "cyan",
    }.get(status, "dim")


@cli.group()
def scenarios():
    """Manage scenario tests (multi-step flows) for a system."""
    pass


@scenarios.command("list")
@click.option(
    "--system",
    "-s",
    "system_identifier",
    required=True,
    help="System slug, name, or UUID",
)
def scenarios_list(system_identifier: str):
    """List scenarios attached to a system."""
    client = _get_client()
    system = _resolve_system(client, system_identifier)
    if not system:
        console.print(f"[red]System not found:[/red] {system_identifier}")
        sys.exit(1)

    try:
        data = client.list_scenarios(system["id"])
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    items = data.get("items", []) if isinstance(data, dict) else []
    if not items:
        console.print("[dim]No scenarios defined for this system.[/dim]")
        return

    table = Table(
        title=f"Scenarios — {system.get('name', system_identifier)}",
        box=box.ROUNDED,
    )
    table.add_column("Name", style="cyan")
    table.add_column("Steps", justify="right")
    table.add_column("Tags", style="magenta")
    table.add_column("ID", style="dim")
    for item in items:
        table.add_row(
            item.get("name", ""),
            str(len(item.get("steps", []))),
            ", ".join(item.get("tags", []) or []),
            str(item.get("id", ""))[:8] + "...",
        )
    console.print(table)


@scenarios.command("run")
@click.option(
    "--system",
    "-s",
    "system_identifier",
    required=True,
    help="System slug, name, or UUID",
)
@click.option(
    "--scenario",
    "-n",
    "scenario_identifier",
    required=True,
    help="Scenario name or UUID",
)
def scenarios_run(system_identifier: str, scenario_identifier: str):
    """Run a single scenario against its system."""
    client = _get_client()
    system = _resolve_system(client, system_identifier)
    if not system:
        console.print(f"[red]System not found:[/red] {system_identifier}")
        sys.exit(1)

    data = client.list_scenarios(system["id"])
    items = data.get("items", []) if isinstance(data, dict) else []
    scenario = next(
        (
            item
            for item in items
            if item.get("name") == scenario_identifier
            or item.get("id") == scenario_identifier
        ),
        None,
    )
    if not scenario:
        console.print(f"[red]Scenario not found:[/red] {scenario_identifier}")
        sys.exit(1)

    console.print(
        f"[cyan]Running scenario[/cyan] [bold]{scenario['name']}[/bold] "
        f"on [magenta]{system.get('name', system_identifier)}[/magenta]..."
    )
    try:
        run = client.run_scenario(system["id"], scenario["id"])
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    status = run.get("status", "unknown")
    style = _scenario_status_style(status)
    console.print(
        f"[{style}]{status.upper()}[/{style}] — "
        f"{run.get('duration_ms', 0)}ms"
    )

    for result in run.get("step_results") or []:
        step_status = result.get("status", "unknown")
        step_style = _scenario_status_style(step_status)
        console.print(
            f"  [{step_style}]{step_status:<8}[/{step_style}] "
            f"{result.get('step_name', '<unnamed>')}"
            f"  ({result.get('duration_ms', 0)}ms)"
        )
        for failure in result.get("failed_assertions") or []:
            assertion = failure.get("assertion", {})
            console.print(
                f"      [red]x[/red] {assertion.get('type')} "
                f"— {failure.get('explanation', '')}"
            )

    if status != "passed":
        sys.exit(1)


@scenarios.command("run-all")
@click.option(
    "--system",
    "-s",
    "system_identifier",
    required=True,
    help="System slug, name, or UUID",
)
@click.option(
    "--fail-on-error",
    is_flag=True,
    help="Exit with code 1 if any scenario fails (CI integration)",
)
def scenarios_run_all(system_identifier: str, fail_on_error: bool):
    """Run every scenario attached to a system.

    With `--fail-on-error`, exits 1 when any scenario fails — suitable for
    wiring into CI pipelines (GitHub Actions, GitLab CI, etc.).
    """
    client = _get_client()
    system = _resolve_system(client, system_identifier)
    if not system:
        console.print(f"[red]System not found:[/red] {system_identifier}")
        sys.exit(1)

    data = client.list_scenarios(system["id"])
    items = data.get("items", []) if isinstance(data, dict) else []
    if not items:
        console.print("[dim]No scenarios to run.[/dim]")
        return

    passed = 0
    failed = 0
    errored = 0
    for scenario in items:
        console.print(f"[cyan]» {scenario['name']}[/cyan]")
        try:
            run = client.run_scenario(system["id"], scenario["id"])
        except Exception as exc:  # noqa: BLE001
            errored += 1
            console.print(f"  [yellow]error[/yellow] {exc}")
            continue

        status = run.get("status", "unknown")
        style = _scenario_status_style(status)
        console.print(
            f"  [{style}]{status}[/{style}] ({run.get('duration_ms', 0)}ms)"
        )
        if status == "passed":
            passed += 1
        elif status == "failed":
            failed += 1
        else:
            errored += 1

        for result in run.get("step_results") or []:
            for failure in result.get("failed_assertions") or []:
                assertion = failure.get("assertion", {})
                console.print(
                    f"    [red]x[/red] {result.get('step_name')}: "
                    f"{assertion.get('type')} — {failure.get('explanation', '')}"
                )

    total = passed + failed + errored
    console.print(
        f"\n[bold]Results:[/bold] "
        f"[green]{passed} passed[/green], "
        f"[red]{failed} failed[/red], "
        f"[yellow]{errored} errored[/yellow] / {total} total"
    )

    if fail_on_error and (failed > 0 or errored > 0):
        sys.exit(1)


# ── invoke ──


@cli.command()
@click.argument("system_identifier")
@click.option("--input", "-i", "input_json", required=True, help="JSON input payload")
def invoke(system_identifier: str, input_json: str):
    """Invoke a multi-agent system by name or ID."""
    import json
    import time

    try:
        payload = json.loads(input_json)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON input:[/red] {e}")
        sys.exit(1)

    client = _get_client()
    system = _resolve_system(client, system_identifier)
    if not system:
        console.print(f"[red]System not found:[/red] {system_identifier}")
        sys.exit(1)

    console.print(
        f"[cyan]Invoking[/cyan] [bold]{system.get('name', system_identifier)}[/bold]..."
    )
    start = time.monotonic()
    try:
        with httpx.Client(timeout=600.0) as http:
            url = f"{client.registry_url}/api/engine/invoke"
            resp = http.post(
                url,
                json={"system_id": system["id"], "input": payload},
                headers=client._headers,
            )
            resp.raise_for_status()
            result = resp.json()
    except Exception as e:
        console.print(f"[red]Invocation failed:[/red] {e}")
        sys.exit(1)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    data = result.get("data", result)
    status = data.get("status", "completed") if isinstance(data, dict) else "completed"
    color = "green" if status == "completed" else "yellow"

    console.print(f"[{color}]{status.upper()}[/{color}] in {elapsed_ms}ms\n")
    console.print_json(json.dumps(data, indent=2, default=str))


# ── status ──


@cli.command()
def status():
    """Show all agents and systems with health summary."""
    client = _get_client()

    try:
        agents_data = client.list_agents(limit=200)
    except Exception as e:
        console.print(f"[red]Error fetching agents:[/red] {e}")
        agents_data = {"items": [], "total": 0}

    try:
        systems_data = client.list_systems(limit=100)
    except Exception as e:
        console.print(f"[red]Error fetching systems:[/red] {e}")
        systems_data = {"items": [], "total": 0}

    # Agents table
    agents = agents_data.get("items", [])
    agent_table = Table(title="Agents", box=box.ROUNDED)
    agent_table.add_column("Name", style="cyan")
    agent_table.add_column("Namespace", style="dim")
    agent_table.add_column("Framework", style="magenta")
    agent_table.add_column("Status")
    agent_table.add_column("Health")

    healthy_count = 0
    for agent in agents:
        st = agent.get("status", "unknown")
        st_style = "green" if st == "active" else "red"
        hlth = agent.get("health", "unknown")
        hlth_style = {"healthy": "green", "slow": "yellow"}.get(hlth, "red")
        if hlth == "healthy":
            healthy_count += 1
        agent_table.add_row(
            agent["name"],
            agent.get("namespace", ""),
            agent.get("framework", ""),
            f"[{st_style}]{st}[/{st_style}]",
            f"[{hlth_style}]{hlth}[/{hlth_style}]",
        )
    console.print(agent_table)

    # Systems table
    systems = systems_data.get("items", [])
    sys_table = Table(title="Systems", box=box.ROUNDED)
    sys_table.add_column("Name", style="cyan")
    sys_table.add_column("Pattern", style="magenta")
    sys_table.add_column("Status")
    sys_table.add_column("Environment", style="dim")

    deployed_count = 0
    for s in systems:
        st = s.get("status", "draft")
        st_style = "green" if st == "deployed" else "dim"
        if st == "deployed":
            deployed_count += 1
        sys_table.add_row(
            s.get("name", ""),
            s.get("pattern", ""),
            f"[{st_style}]{st}[/{st_style}]",
            s.get("environment", ""),
        )
    console.print(sys_table)

    total_agents = len(agents)
    total_systems = len(systems)
    console.print(
        f"\n[bold]{total_agents} agents[/bold] ({healthy_count} healthy) "
        f"[dim]\u00b7[/dim] "
        f"[bold]{total_systems} systems[/bold] ({deployed_count} deployed)"
    )


# ── logs ──


@cli.command()
@click.argument("agent_name")
@click.option("--follow", "-f", is_flag=True, help="Poll health every 2s")
@click.option("--tail", "-t", "tail_count", default=50, help="Number of recent invocations")
def logs(agent_name: str, follow: bool, tail_count: int):
    """Show agent health and invocation stats."""
    import time

    client = _get_client()
    try:
        results = client.search_agents(agent_name)
        items = results.get("items", [])
        agent = next((a for a in items if a["name"] == agent_name), None) or (
            items[0] if items else None
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not agent:
        console.print(f"[red]Agent not found:[/red] {agent_name}")
        sys.exit(1)

    endpoint = agent.get("a2a_endpoint", "")
    console.print(f"[bold]{agent['name']}[/bold] ({agent.get('namespace', '')})")
    console.print(f"[dim]Endpoint:[/dim] {endpoint or 'N/A'}")

    def _print_health() -> str:
        try:
            h = client.check_health(agent["id"])
            st = h.get("status", "unknown")
            color = {"healthy": "green", "slow": "yellow"}.get(st, "red")
            latency = h.get("latency_ms", "?")
            console.print(f"  [{color}]{st}[/{color}] ({latency}ms)")
            return st
        except Exception:
            console.print("  [red]unreachable[/red]")
            return "unreachable"

    console.print("[bold]Health:[/bold]")
    _print_health()

    # Show recent invocation stats via engine
    try:
        with httpx.Client(timeout=10.0) as http:
            url = f"{client.registry_url}/api/engine/invocations"
            resp = http.get(
                url,
                params={"agent_name": agent_name, "limit": tail_count},
                headers=client._headers,
            )
            if resp.status_code == 200:
                inv_data = resp.json().get("data", {})
                inv_items = inv_data.get("items", []) if isinstance(inv_data, dict) else []
                if inv_items:
                    inv_table = Table(
                        title=f"Recent Invocations (last {tail_count})", box=box.SIMPLE
                    )
                    inv_table.add_column("ID", style="dim")
                    inv_table.add_column("Status")
                    inv_table.add_column("Duration")
                    for inv in inv_items[:tail_count]:
                        inv_st = inv.get("status", "unknown")
                        inv_color = "green" if inv_st == "completed" else "red"
                        inv_table.add_row(
                            str(inv.get("id", ""))[:8] + "...",
                            f"[{inv_color}]{inv_st}[/{inv_color}]",
                            f"{inv.get('duration_ms', '?')}ms",
                        )
                    console.print(inv_table)
    except Exception:
        pass  # Invocation history is best-effort

    if follow:
        console.print("\n[dim]Polling health every 2s (Ctrl+C to stop)...[/dim]")
        last_status = ""
        try:
            while True:
                time.sleep(2)
                current = _print_health()
                if current != last_status and last_status:
                    console.print(
                        f"  [yellow]Status changed:[/yellow] {last_status} -> {current}"
                    )
                last_status = current
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")


if __name__ == "__main__":
    cli()
