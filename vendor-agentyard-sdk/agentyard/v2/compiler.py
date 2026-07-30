"""Topology compiler for AgentYard v2.

Reads a system spec (nodes + edges + pattern) plus per-agent capability cards
and emits a per-node :class:`NodeConfig` that the v2 runtime drops at
``/yard/config.yaml`` to drive its adaptive transports.

The compiler is the bridge between Layer 2 (system topology) and Layer 4 (the
adaptive SDK runtime). Agents stay pure functions; this module decides which
input/output transport each node should use, what URL it forwards to, what
topics it subscribes to, what memory contract it must honour, and which
resources are bound where.

Six patterns are supported. Each gets its own compile function. They share a
small toolkit (``_index_edges``, ``_endpoint_for``, ``_emit_target``) and a
single :func:`compile_topology` entrypoint that dispatches on
``SystemSpecLite.pattern``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx
import yaml

logger = logging.getLogger("yard.compiler")


SUPPORTED_PATTERNS = frozenset({
    "sequential",
    "fanout",
    "dag",
    "streaming",
    "event_driven",
    "saga",
})

# Map legacy engine_type / vendor-neutral pattern names into the 6 patterns the
# compiler knows how to wire. Anything outside this map falls back to "dag".
_PATTERN_ALIASES: dict[str, str] = {
    "sequential": "sequential",
    "chain_executor": "sequential",
    "parallel_fanout": "fanout",
    "fanout_executor": "fanout",
    "fanout": "fanout",
    "evaluator_loop": "dag",
    "rule_router": "dag",
    "supervisor": "dag",
    "llm_supervisor": "dag",
    "debate": "dag",
    "dag": "dag",
    "dag_executor": "dag",
    "saga": "saga",
    "saga_executor": "saga",
    "streaming": "streaming",
    "event_driven": "event_driven",
}


# ---------- Inputs ----------

@dataclass(frozen=True)
class AgentCapability:
    """The subset of an agent card the compiler needs."""
    name: str
    namespace: str = ""
    intent: str = ""
    needs: tuple[dict, ...] = field(default_factory=tuple)
    memory: dict = field(default_factory=dict)
    behavior: dict = field(default_factory=dict)
    input_schema: dict | None = None
    output_schema: dict | None = None
    supports_adaptive_transport: bool = False
    """True if the agent card advertises a ``transport`` block — i.e. it's a v2
    SDK agent that understands stream_consume / subscribe / aggregate / emit.
    Non-v2 agents (LangChain/CrewAI wrapped in FastAPI, raw A2A HTTP services)
    lack this and must be forced to http+sync regardless of the system's mode."""

    @classmethod
    def from_card(cls, card: dict) -> "AgentCapability":
        # A v2 SDK agent card has ``transport: {mode, input, output, audit_enabled}``.
        # Any other card (legacy, LangChain wrapper, hand-rolled A2A) omits it or
        # has a scalar placeholder — treat as HTTP-only.
        transport = card.get("transport")
        adaptive = isinstance(transport, dict) and bool(
            transport.get("input") or transport.get("output") or transport.get("mode")
        )
        return cls(
            name=str(card.get("name", "")),
            namespace=str(card.get("namespace", "")),
            intent=str(card.get("intent", "")),
            needs=tuple(card.get("needs") or ()),
            memory=dict(card.get("memory") or {}),
            behavior=dict(card.get("behavior") or {}),
            input_schema=card.get("input_schema"),
            output_schema=card.get("output_schema"),
            supports_adaptive_transport=adaptive,
        )


@dataclass(frozen=True)
class SystemNodeLite:
    id: str
    agent_name: str
    endpoint: str = ""
    config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SystemEdgeLite:
    source_node_id: str
    target_node_id: str
    condition: dict | None = None
    transform: dict = field(default_factory=dict)


TransportMode = str  # Literal["centralized", "mesh", "hybrid"]

SUPPORTED_TRANSPORT_MODES = frozenset({"centralized", "mesh", "hybrid"})


@dataclass(frozen=True)
class SystemSpecLite:
    """Compiler input — the minimum needed to wire a topology."""
    id: str
    name: str
    namespace: str
    pattern: str
    nodes: tuple[SystemNodeLite, ...]
    edges: tuple[SystemEdgeLite, ...]
    resources: dict = field(default_factory=dict)
    """System-level resource bindings (e.g. {"postgres": "postgresql://..."})."""
    transport_mode: TransportMode = "centralized"
    """One of 'centralized' | 'mesh' | 'hybrid'. See apply_transport_mode."""

    @classmethod
    def from_dict(cls, system: dict) -> "SystemSpecLite":
        nodes = tuple(
            SystemNodeLite(
                id=str(n.get("id", "")),
                agent_name=str(n.get("agent_name") or n.get("name") or n.get("agent_id", "")),
                endpoint=str(n.get("endpoint", "")),
                config=dict(n.get("config") or {}),
            )
            for n in (system.get("nodes") or [])
        )
        edges = tuple(
            SystemEdgeLite(
                source_node_id=str(e.get("source_node_id") or e.get("source", "")),
                target_node_id=str(e.get("target_node_id") or e.get("target", "")),
                condition=e.get("condition"),
                transform=dict(e.get("transform") or {}),
            )
            for e in (system.get("edges") or [])
        )
        orchestrator_cfg = system.get("orchestrator_config") or {}
        transport_mode = str(
            system.get("transport_mode")
            or orchestrator_cfg.get("transport_mode")
            or "centralized"
        )
        if transport_mode not in SUPPORTED_TRANSPORT_MODES:
            transport_mode = "centralized"
        raw_pattern = str(
            (system.get("orchestrator_config") or {}).get("engine_type")
            or system.get("pattern")
            or "dag"
        )
        pattern = _PATTERN_ALIASES.get(raw_pattern, "dag")
        return cls(
            id=str(system.get("id", "")),
            name=str(system.get("name", "system")),
            namespace=str(system.get("namespace", "default")),
            pattern=pattern,
            nodes=nodes,
            edges=edges,
            resources=dict(system.get("resources") or {}),
            transport_mode=transport_mode,
        )


# ---------- Output ----------

@dataclass
class NodeConfig:
    """One node's compiled runtime config — written to /yard/config.yaml."""
    system_id: str
    node_id: str
    input_mode: str = "http"
    output_mode: str = "sync"
    downstream_url: str = ""
    upstream_url: str = ""
    emit_targets: list[str] = field(default_factory=list)
    subscribe_topics: list[str] = field(default_factory=list)
    input_stream: str = ""
    input_group: str = ""
    callback_url: str = ""
    aggregate_batch_size: int = 5
    aggregate_window_seconds: float = 5.0
    memory: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    failure: dict = field(default_factory=dict)
    audit_stream: str = ""
    """Hybrid mode: Redis stream the SDK mirrors each result to, in parallel with
    the sync response. Empty when auditing is off."""
    transport_mode: str = "centralized"
    """Effective transport mode for this node — surfaced for Mission Control UI."""
    notes: list[str] = field(default_factory=list)
    """Compiler diagnostics — what was decided and why."""

    def to_yaml(self) -> str:
        data: dict[str, Any] = {
            "system_id": self.system_id,
            "node_id": self.node_id,
            "transport_mode": self.transport_mode,
            "input_mode": self.input_mode,
            "output_mode": self.output_mode,
            "downstream_url": self.downstream_url,
            "upstream_url": self.upstream_url,
            "emit_targets": self.emit_targets,
            "subscribe_topics": self.subscribe_topics,
            "input_stream": self.input_stream,
            "input_group": self.input_group,
            "callback_url": self.callback_url,
            "audit_stream": self.audit_stream,
            "memory": self.memory,
            "resources": self.resources,
            "failure": self.failure,
        }
        if self.input_mode == "aggregate":
            data["aggregate_batch_size"] = self.aggregate_batch_size
            data["aggregate_window_seconds"] = self.aggregate_window_seconds
        cleaned = {k: v for k, v in data.items() if v not in ("", [], {}, None)}
        return yaml.safe_dump(cleaned, sort_keys=False, default_flow_style=False)


# ---------- Helpers ----------

def _index_edges(edges: Iterable[SystemEdgeLite]) -> tuple[dict[str, list[SystemEdgeLite]], dict[str, list[SystemEdgeLite]]]:
    """Build outgoing + incoming edge indices keyed by node id."""
    outgoing: dict[str, list[SystemEdgeLite]] = defaultdict(list)
    incoming: dict[str, list[SystemEdgeLite]] = defaultdict(list)
    for e in edges:
        outgoing[e.source_node_id].append(e)
        incoming[e.target_node_id].append(e)
    return outgoing, incoming


def _endpoint_for(spec: SystemSpecLite, node_id: str) -> str:
    """Resolve the HTTP endpoint URL for a node by id."""
    for n in spec.nodes:
        if n.id == node_id:
            return n.endpoint
    return ""


def _emit_target(system_id: str, source_node: str, target_node: str, *, channel: bool = False) -> str:
    """Standard naming for inter-node Redis streams/channels."""
    prefix = "channel:" if channel else "stream:"
    return f"{prefix}yard:sys:{system_id}:edge:{source_node}:{target_node}"


def _resolve_resources(cap: AgentCapability | None, system_resources: dict) -> dict:
    """Pick out only the resources the agent declared needing.

    System-level resources is a free-form dict (e.g.
    ``{"postgres": {"url": ...}, "llm": {...}}``). We project it down to the
    keys the agent's needs declare so secrets/connection strings don't leak
    to nodes that don't ask for them.
    """
    if not cap:
        return {}
    resolved: dict[str, Any] = {}
    declared_kinds = {n.get("kind") for n in cap.needs if n.get("kind")}
    for kind in declared_kinds:
        if kind in system_resources:
            resolved[kind] = system_resources[kind]
    return resolved


def _build_failure(cap: AgentCapability | None) -> dict:
    if not cap:
        return {}
    behavior = cap.behavior or {}
    if behavior.get("is_idempotent"):
        return {"mode": "retry", "max_retries": 3}
    return {"mode": "abort"}


# ---------- Per-pattern compilers ----------

def _compile_sequential(spec: SystemSpecLite, caps: dict[str, AgentCapability | None]) -> dict[str, NodeConfig]:
    """A → B → C: each node forwards its result to the next via HTTP, last
    node returns sync to the caller. We use http+stream so the chain runs as
    forward-only RPCs without the orchestrator round-tripping every result."""
    out, _ = _index_edges(spec.edges)
    cfgs: dict[str, NodeConfig] = {}
    for node in spec.nodes:
        cap = caps.get(node.agent_name)
        cfg = NodeConfig(
            system_id=spec.id,
            node_id=node.id,
            input_mode="http",
            memory=cap.memory if cap else {},
            resources=_resolve_resources(cap, spec.resources),
            failure=_build_failure(cap),
        )
        outs = out.get(node.id, [])
        if outs:
            target = outs[0].target_node_id
            cfg.output_mode = "stream"
            cfg.downstream_url = _endpoint_for(spec, target)
            cfg.notes.append(f"sequential: forwards to {target}")
        else:
            cfg.output_mode = "sync"
            cfg.notes.append("sequential: terminal node returns sync")
        cfgs[node.id] = cfg
    return cfgs


def _compile_fanout(spec: SystemSpecLite, caps: dict[str, AgentCapability | None]) -> dict[str, NodeConfig]:
    """Source → many workers → merger.

    - source: http + emit (publishes to N streams, one per worker)
    - workers: stream_consume + emit (consume one stream, emit to merger)
    - merger: aggregate + sync (buffer N inputs, dispatch as a batch)

    The merger is identified as any node with in-degree > 1.
    """
    out, inc = _index_edges(spec.edges)
    cfgs: dict[str, NodeConfig] = {}
    for node in spec.nodes:
        cap = caps.get(node.agent_name)
        cfg = NodeConfig(
            system_id=spec.id,
            node_id=node.id,
            memory=cap.memory if cap else {},
            resources=_resolve_resources(cap, spec.resources),
            failure=_build_failure(cap),
        )
        in_degree = len(inc.get(node.id, []))
        out_degree = len(out.get(node.id, []))

        if out_degree > 1:
            cfg.input_mode = "http"
            cfg.output_mode = "emit"
            cfg.emit_targets = [
                _emit_target(spec.id, node.id, e.target_node_id) for e in out[node.id]
            ]
            cfg.notes.append(f"fanout: source emits to {len(out[node.id])} workers")
        elif in_degree > 1:
            cfg.input_mode = "aggregate"
            cfg.output_mode = "sync"
            cfg.aggregate_batch_size = in_degree
            cfg.aggregate_window_seconds = 30.0
            cfg.notes.append(f"fanout: merger aggregates {in_degree} inputs")
        elif in_degree == 1 and out_degree == 1:
            edge_in = inc[node.id][0]
            edge_out = out[node.id][0]
            cfg.input_mode = "stream_consume"
            cfg.input_stream = _emit_target(spec.id, edge_in.source_node_id, node.id, channel=False).removeprefix("stream:")
            cfg.input_group = f"yard-{node.id}"
            cfg.output_mode = "emit"
            cfg.emit_targets = [_emit_target(spec.id, node.id, edge_out.target_node_id)]
            cfg.notes.append("fanout: worker consumes 1 stream, emits to merger")
        else:
            # Lonely node — fall back to sync
            cfg.input_mode = "http"
            cfg.output_mode = "sync"
            cfg.notes.append("fanout: standalone node, sync fallback")
        cfgs[node.id] = cfg
    return cfgs


def _compile_dag(spec: SystemSpecLite, caps: dict[str, AgentCapability | None]) -> dict[str, NodeConfig]:
    """General DAG: each non-terminal node forwards to next via http+stream;
    nodes with multiple outgoing edges emit; nodes with multiple incoming
    edges aggregate. Terminal node returns sync."""
    out, inc = _index_edges(spec.edges)
    cfgs: dict[str, NodeConfig] = {}
    for node in spec.nodes:
        cap = caps.get(node.agent_name)
        cfg = NodeConfig(
            system_id=spec.id,
            node_id=node.id,
            memory=cap.memory if cap else {},
            resources=_resolve_resources(cap, spec.resources),
            failure=_build_failure(cap),
        )
        in_degree = len(inc.get(node.id, []))
        out_degree = len(out.get(node.id, []))

        # Input mode
        if in_degree > 1:
            cfg.input_mode = "aggregate"
            cfg.aggregate_batch_size = in_degree
            cfg.notes.append(f"dag: {in_degree} incoming edges → aggregate")
        else:
            cfg.input_mode = "http"

        # Output mode
        if out_degree == 0:
            cfg.output_mode = "sync"
            cfg.notes.append("dag: terminal → sync")
        elif out_degree == 1:
            cfg.output_mode = "stream"
            cfg.downstream_url = _endpoint_for(spec, out[node.id][0].target_node_id)
            cfg.notes.append(f"dag: 1 outgoing → stream to {out[node.id][0].target_node_id}")
        else:
            cfg.output_mode = "emit"
            cfg.emit_targets = [
                _emit_target(spec.id, node.id, e.target_node_id) for e in out[node.id]
            ]
            cfg.notes.append(f"dag: {out_degree} outgoing → emit")
        cfgs[node.id] = cfg
    return cfgs


def _compile_streaming(spec: SystemSpecLite, caps: dict[str, AgentCapability | None]) -> dict[str, NodeConfig]:
    """True pipeline: every node consumes from a Redis stream and emits to
    the next. No HTTP forwarding overhead. The first node has http input as
    its 'kick-off' entrypoint."""
    out, inc = _index_edges(spec.edges)
    cfgs: dict[str, NodeConfig] = {}
    for node in spec.nodes:
        cap = caps.get(node.agent_name)
        cfg = NodeConfig(
            system_id=spec.id,
            node_id=node.id,
            memory=cap.memory if cap else {},
            resources=_resolve_resources(cap, spec.resources),
            failure=_build_failure(cap),
        )
        ins = inc.get(node.id, [])
        outs = out.get(node.id, [])

        if ins:
            cfg.input_mode = "stream_consume"
            cfg.input_stream = f"yard:sys:{spec.id}:node:{node.id}:in"
            cfg.input_group = f"yard-{node.id}"
            cfg.notes.append("streaming: consume from in-stream")
        else:
            cfg.input_mode = "http"
            cfg.notes.append("streaming: HTTP kickoff (head node)")

        if outs:
            cfg.output_mode = "emit"
            cfg.emit_targets = [
                f"stream:yard:sys:{spec.id}:node:{e.target_node_id}:in" for e in outs
            ]
            cfg.notes.append(f"streaming: emit to {len(outs)} downstream stream(s)")
        else:
            cfg.output_mode = "emit"
            cfg.emit_targets = [f"stream:yard:sys:{spec.id}:results"]
            cfg.notes.append("streaming: terminal → results stream")
        cfgs[node.id] = cfg
    return cfgs


def _compile_event_driven(spec: SystemSpecLite, caps: dict[str, AgentCapability | None]) -> dict[str, NodeConfig]:
    """Pub/sub topology — nodes communicate via Redis channels rather than
    direct addressing.

    - Producers (no incoming edges): http + emit to channels
    - Consumers (have incoming edges): subscribe + emit on results

    Channel naming: ``yard:topic:{system_id}:{source}:{target}``.
    """
    out, inc = _index_edges(spec.edges)
    cfgs: dict[str, NodeConfig] = {}
    for node in spec.nodes:
        cap = caps.get(node.agent_name)
        cfg = NodeConfig(
            system_id=spec.id,
            node_id=node.id,
            memory=cap.memory if cap else {},
            resources=_resolve_resources(cap, spec.resources),
            failure=_build_failure(cap),
        )
        ins = inc.get(node.id, [])
        outs = out.get(node.id, [])

        if ins:
            cfg.input_mode = "subscribe"
            cfg.subscribe_topics = [
                f"yard:topic:{spec.id}:{e.source_node_id}:{node.id}" for e in ins
            ]
            cfg.notes.append(f"event_driven: subscribe to {len(ins)} topic(s)")
        else:
            cfg.input_mode = "http"
            cfg.notes.append("event_driven: HTTP entrypoint (producer)")

        if outs:
            cfg.output_mode = "emit"
            cfg.emit_targets = [
                f"channel:yard:topic:{spec.id}:{node.id}:{e.target_node_id}" for e in outs
            ]
            cfg.notes.append(f"event_driven: emit to {len(outs)} topic(s)")
        else:
            cfg.output_mode = "emit"
            cfg.emit_targets = [f"channel:yard:topic:{spec.id}:results"]
            cfg.notes.append("event_driven: terminal → results channel")
        cfgs[node.id] = cfg
    return cfgs


def _compile_saga(spec: SystemSpecLite, caps: dict[str, AgentCapability | None]) -> dict[str, NodeConfig]:
    """Sequential with compensation: each step is http+sync (the orchestrator
    drives compensation), and the failure policy declares a compensation
    agent if the node config carries one (``config.compensation_agent``)."""
    cfgs: dict[str, NodeConfig] = {}
    for node in spec.nodes:
        cap = caps.get(node.agent_name)
        compensation = (node.config or {}).get("compensation_agent", "")
        failure = _build_failure(cap)
        if compensation:
            failure = {
                "mode": "compensate",
                "fallback_agent": compensation,
                "max_retries": 1,
            }
        cfg = NodeConfig(
            system_id=spec.id,
            node_id=node.id,
            input_mode="http",
            output_mode="sync",
            memory=cap.memory if cap else {},
            resources=_resolve_resources(cap, spec.resources),
            failure=failure,
        )
        cfg.notes.append(
            f"saga: sync step{' with compensation=' + compensation if compensation else ''}"
        )
        cfgs[node.id] = cfg
    return cfgs


_PATTERN_COMPILERS = {
    "sequential": _compile_sequential,
    "fanout": _compile_fanout,
    "dag": _compile_dag,
    "streaming": _compile_streaming,
    "event_driven": _compile_event_driven,
    "saga": _compile_saga,
}


# Patterns where one orchestration model is obviously right. Used by
# :func:`suggest_transport_mode` to nudge the user in Studio.
_MESH_NATURAL_PATTERNS = frozenset({"streaming", "event_driven"})
_CENTRALIZED_NATURAL_PATTERNS = frozenset({
    "evaluator_loop", "rule_router", "supervisor", "debate", "saga",
})


def suggest_transport_mode(pattern: str) -> str:
    """Recommend a transport mode given the raw pattern/engine type.

    - streaming / event_driven → mesh (centralized adds pointless latency)
    - evaluator_loop / rule_router / supervisor / debate / saga → centralized
      (the orchestrator needs to own control flow)
    - everything else → centralized (safe default; user can opt into mesh)
    """
    normalized = _PATTERN_ALIASES.get(pattern, pattern)
    if normalized in _MESH_NATURAL_PATTERNS:
        return "mesh"
    if pattern in _CENTRALIZED_NATURAL_PATTERNS or normalized in _CENTRALIZED_NATURAL_PATTERNS:
        return "centralized"
    return "centralized"


def _force_http_sync(cfg: NodeConfig, reason: str) -> None:
    """Downgrade a single node to http+sync transport, preserving semantics."""
    cfg.input_mode = "http"
    cfg.output_mode = "sync"
    cfg.downstream_url = ""
    cfg.upstream_url = ""
    cfg.emit_targets = []
    cfg.subscribe_topics = []
    cfg.input_stream = ""
    cfg.input_group = ""
    cfg.callback_url = ""
    cfg.notes.append(reason)


def apply_transport_mode(
    cfgs: dict[str, NodeConfig],
    mode: str,
    spec: SystemSpecLite,
    caps: dict[str, AgentCapability | None] | None = None,
) -> dict[str, NodeConfig]:
    """Post-process compiled configs according to the system's transport mode.

    - ``centralized``: downgrade all transport fields to HTTP+sync so an
      external orchestrator (LangGraph / CrewAI / Temporal / generated
      Python) drives every hop. Memory ACL, resource bindings, and failure
      policy are preserved — those are framework-agnostic.
    - ``mesh``: pass through unchanged. The per-pattern compiler's choices
      stand; agents forward/emit/subscribe to each other directly.
    - ``hybrid``: centralized transports plus a shared ``audit_stream`` that
      the SDK mirrors each result to in parallel with the sync response.
      Gives you the orchestrator-driven mental model with a distributed
      event log for replay, debugging, and compliance.

    The ``spec`` is passed in so hybrid can name the audit stream.
    """
    if mode not in SUPPORTED_TRANSPORT_MODES:
        logger.warning("unknown_transport_mode fallback=centralized mode=%s", mode)
        mode = "centralized"

    # Stamp the mode on every node for UI/telemetry — regardless of whether
    # we change transport fields below.
    for cfg in cfgs.values():
        cfg.transport_mode = mode

    # Index nodes → agent_name so we can consult capabilities per-node.
    node_to_agent = {n.id: n.agent_name for n in spec.nodes}
    caps = caps or {}

    if mode == "mesh":
        # Mesh mode needs a rendezvous point where the kicker orchestrator
        # can pick up terminal results. Any node with no outgoing edges
        # that isn't already emitting (i.e. would return sync) gets
        # redirected to the standard results stream.
        results_stream = f"yard:sys:{spec.id}:results"
        out_edges_by_node: dict[str, int] = {n.id: 0 for n in spec.nodes}
        for e in spec.edges:
            if e.source_node_id in out_edges_by_node:
                out_edges_by_node[e.source_node_id] += 1

        for node_id, cfg in cfgs.items():
            agent_name = node_to_agent.get(node_id, "")
            cap = caps.get(agent_name)

            # Step 1: capability-aware fallback — non-v2 agents go http+sync.
            if cap is None:
                _force_http_sync(
                    cfg,
                    f"mesh→http+sync: couldn't discover capabilities for {agent_name!r}"
                    " (conservative fallback)",
                )
            elif not cap.supports_adaptive_transport:
                _force_http_sync(
                    cfg,
                    f"mesh→http+sync: {agent_name!r} is not a v2 SDK agent; "
                    "other nodes in this mesh keep their compiled transports",
                )

            # Step 2: terminal v2 nodes emit to the results stream so the
            # mesh kicker can collect them. A "terminal" node has no
            # outgoing edges. Skip if the per-pattern compiler already set
            # emit_targets (e.g. streaming pattern already emits to results).
            is_terminal = out_edges_by_node.get(node_id, 0) == 0
            already_emitting_results = any(
                results_stream in t for t in cfg.emit_targets
            )
            if (
                is_terminal
                and cap is not None
                and cap.supports_adaptive_transport
                and not already_emitting_results
            ):
                cfg.output_mode = "emit"
                cfg.downstream_url = ""
                cfg.emit_targets = [f"stream:{results_stream}"]
                cfg.notes.append(
                    f"mesh: terminal node redirected to results stream "
                    f"({results_stream}) for kicker pickup"
                )
        return cfgs

    audit_stream = f"yard:sys:{spec.id}:audit" if mode == "hybrid" else ""

    for cfg in cfgs.values():
        # centralized / hybrid: everyone is http+sync so the orchestrator drives.
        cfg.input_mode = "http"
        cfg.output_mode = "sync"
        cfg.downstream_url = ""
        cfg.upstream_url = ""
        cfg.emit_targets = []
        cfg.subscribe_topics = []
        cfg.input_stream = ""
        cfg.input_group = ""
        cfg.callback_url = ""
        if mode == "hybrid":
            # Hybrid audit only works on v2 SDK agents — they're the only ones
            # that read audit_stream from /yard/config.yaml. Non-v2 agents just
            # ignore the field, which is fine.
            agent_name = node_to_agent.get(cfg.node_id, "")
            cap = caps.get(agent_name)
            if cap is not None and cap.supports_adaptive_transport:
                cfg.audit_stream = audit_stream
                cfg.notes.append("hybrid: orchestrator-driven + audit mirror")
            else:
                cfg.audit_stream = ""
                cfg.notes.append(
                    "hybrid: orchestrator-driven (audit skipped — "
                    "not a v2 SDK agent)"
                )
        else:
            cfg.audit_stream = ""
            cfg.notes.append(
                "centralized: transport downgraded to http+sync so the "
                "orchestrator drives every hop"
            )
    return cfgs


# ---------- Public API ----------

def compile_topology(spec: SystemSpecLite, caps: dict[str, AgentCapability | None]) -> dict[str, NodeConfig]:
    """Turn a system spec + agent capabilities into a per-node config map.

    Args:
        spec: System topology (nodes, edges, pattern, transport_mode).
        caps: Map of agent_name → AgentCapability (None entries mean the
            capability couldn't be fetched; defaults are used).

    Returns:
        Mapping of node_id → NodeConfig. Empty if the spec has no nodes.

    The caller doesn't need to invoke :func:`apply_transport_mode` — this
    function does so automatically using ``spec.transport_mode``.
    """
    if not spec.nodes:
        return {}
    pattern = _PATTERN_ALIASES.get(spec.pattern, spec.pattern)
    compiler = _PATTERN_COMPILERS.get(pattern)
    if not compiler:
        logger.warning("unknown_pattern fallback=dag pattern=%s", spec.pattern)
        compiler = _compile_dag
    cfgs = compiler(spec, caps)
    return apply_transport_mode(cfgs, spec.transport_mode, spec, caps=caps)


_CAPABILITY_CACHE: dict[str, tuple[float, AgentCapability | None]] = {}
_CAPABILITY_CACHE_TTL_SECONDS = 60.0


def _cache_get(key: str) -> AgentCapability | None | _Sentinel:
    """Return cached cap, or _MISS if not present / expired."""
    import time as _time
    entry = _CAPABILITY_CACHE.get(key)
    if entry is None:
        return _MISS
    expires_at, value = entry
    if _time.monotonic() > expires_at:
        _CAPABILITY_CACHE.pop(key, None)
        return _MISS
    return value


def _cache_put(key: str, value: AgentCapability | None) -> None:
    import time as _time
    _CAPABILITY_CACHE[key] = (
        _time.monotonic() + _CAPABILITY_CACHE_TTL_SECONDS,
        value,
    )


class _Sentinel:
    pass


_MISS = _Sentinel()


def invalidate_capability_cache(agent_name: str | None = None) -> None:
    """Clear one entry or the whole cache. Call after agent re-register."""
    if agent_name is None:
        _CAPABILITY_CACHE.clear()
    else:
        for k in list(_CAPABILITY_CACHE):
            if k.startswith(f"{agent_name}|"):
                _CAPABILITY_CACHE.pop(k, None)


async def fetch_capabilities(
    nodes: Iterable[SystemNodeLite], *, timeout: float = 5.0
) -> dict[str, AgentCapability | None]:
    """Fetch /.well-known/agent.json for every unique node endpoint.

    Results are cached in-process for 60s keyed by ``agent_name|endpoint``
    so the topology endpoint doesn't ping every agent on every request.
    Failures are also cached (as ``None``) to avoid retry storms when an
    agent is temporarily down; TTL-based expiry recovers automatically.

    Returns a map keyed by agent_name.
    """
    seen: dict[str, str] = {}
    for n in nodes:
        if n.agent_name and n.endpoint and n.agent_name not in seen:
            seen[n.agent_name] = n.endpoint
    out: dict[str, AgentCapability | None] = {}
    if not seen:
        return out

    to_fetch: dict[str, str] = {}
    for name, endpoint in seen.items():
        cache_key = f"{name}|{endpoint}"
        cached = _cache_get(cache_key)
        if isinstance(cached, _Sentinel):
            to_fetch[name] = endpoint
        else:
            out[name] = cached

    if not to_fetch:
        return out

    async with httpx.AsyncClient(timeout=timeout) as client:
        for name, endpoint in to_fetch.items():
            url = endpoint.rstrip("/") + "/.well-known/agent.json"
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                cap = AgentCapability.from_card(resp.json())
                out[name] = cap
                _cache_put(f"{name}|{endpoint}", cap)
            except Exception as exc:
                logger.warning("fetch_capability_failed agent=%s url=%s err=%s", name, url, exc)
                out[name] = None
                _cache_put(f"{name}|{endpoint}", None)
    return out


def explain_compilation(cfgs: dict[str, NodeConfig]) -> str:
    """Human-readable summary of what the compiler decided per node."""
    lines: list[str] = []
    for node_id, cfg in cfgs.items():
        lines.append(f"[{node_id}] {cfg.input_mode} → {cfg.output_mode}")
        for note in cfg.notes:
            lines.append(f"    • {note}")
    return "\n".join(lines)
