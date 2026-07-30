"""Structured logging for agents — feeds into AgentYard monitoring."""

import json
import os
import sys
from datetime import datetime, timezone


class YardLogger:
    """Logger that outputs structured JSON, compatible with AgentYard log collection."""

    def __init__(
        self, agent_name: str, node_id: str = "", system_id: str = ""
    ) -> None:
        self.agent_name = agent_name
        self.node_id = node_id
        self.system_id = system_id

    def _log(self, level: str, msg: str, **extra: object) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "agent": self.agent_name,
            "node_id": self.node_id,
            "system_id": self.system_id,
            "msg": msg,
            **extra,
        }
        print(json.dumps(entry), file=sys.stderr, flush=True)

    def debug(self, msg: str, **kw: object) -> None:
        self._log("debug", msg, **kw)

    def info(self, msg: str, **kw: object) -> None:
        self._log("info", msg, **kw)

    def warning(self, msg: str, **kw: object) -> None:
        self._log("warning", msg, **kw)

    def error(self, msg: str, **kw: object) -> None:
        self._log("error", msg, **kw)


def get_logger(agent_name: str = "") -> YardLogger:
    return YardLogger(
        agent_name=agent_name or os.environ.get("YARD_AGENT_NAME", "unknown"),
        node_id=os.environ.get("YARD_NODE_ID", ""),
        system_id=os.environ.get("YARD_SYSTEM_ID", ""),
    )
