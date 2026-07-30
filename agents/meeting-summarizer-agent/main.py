"""Meeting Summarizer Agent — summarizes meeting transcripts.

v2 SDK: pure business logic. Extracts action items, decisions, and key points
from meeting transcripts.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentyard.v2 import MemoryContract, Resource, yard


def _system_prompt(meeting_type: str) -> str:
    return (
        f"You are an executive assistant summarizing a {meeting_type} meeting. "
        "Produce a structured summary. Return ONLY valid JSON with: summary (2-3 sentence overview), "
        "key_points (array of strings), action_items (array of {action, owner, deadline, status}), "
        "decisions (array of strings), participants (array of names), meeting_type, "
        "stats (word_count, duration_estimate, action_items_count, decisions_count), "
        "follow_up_date (suggested next meeting date or null)."
    )


class Input(BaseModel):
    transcript: str
    meeting_type: Literal["standup", "board", "sales", "general"] = "general"


class Output(BaseModel):
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    action_items: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    meeting_type: str = "general"
    stats: dict[str, Any] = Field(default_factory=dict)
    follow_up_date: str | None = None


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


@yard.agent(
    name="meeting-summarizer-agent",
    namespace="acme/productivity",
    intent="Summarize meeting transcripts into key points, decisions, and action items",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(reads=[], writes=["meeting_summary"]),
    port=9045,
)
async def meeting_summarizer(input: Input, ctx) -> Output:
    """Produce a structured meeting summary with action items and decisions."""
    prompt = (
        f"{_system_prompt(input.meeting_type)}\n\n"
        f"Meeting transcript:\n{input.transcript[:8000]}"
    )
    llm_text = await ctx.llm.complete(prompt)
    parsed = _parse_json(llm_text)

    result = Output(
        summary=parsed.get("summary", ""),
        key_points=parsed.get("key_points") or [],
        action_items=parsed.get("action_items") or [],
        decisions=parsed.get("decisions") or [],
        participants=parsed.get("participants") or [],
        meeting_type=parsed.get("meeting_type") or input.meeting_type,
        stats=parsed.get("stats") or {},
        follow_up_date=parsed.get("follow_up_date"),
    )
    ctx.memory["meeting_summary"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
