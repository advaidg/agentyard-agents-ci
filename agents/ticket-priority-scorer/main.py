"""Ticket Priority Scorer — scores a support ticket's handling priority.

v2 SDK, Bedrock-backed: pure business logic only. Transport, retries, and
infrastructure are handled by the runtime. Built from scratch this session
as an end-to-end proof of the documented v2 + Bedrock path (SDK_GUIDE.md),
deployed standalone to EKS, and composed into a new system afterward.

Chains naturally after triage-classifier (accepts its topic/urgency/message
output) but also works standalone on a bare message.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel

from agentyard.v2 import MemoryContract, Resource, yard

logger = logging.getLogger("ticket-priority-scorer")

SCORE_PROMPT = """You are a customer support operations lead deciding how urgently a ticket needs a human to look at it.

Ticket:
\"\"\"
{message}
\"\"\"

Known topic: {topic}
Known urgency signal: {urgency}

Respond with a JSON object containing:
- "priority": an integer 1-5 (5 = drop everything, 1 = whenever)
- "reasoning": one sentence explaining the score

Respond with ONLY the JSON object, no markdown, no commentary."""


class Input(BaseModel):
    message: str
    topic: str | None = None
    urgency: str | None = None
    customer_id: str | None = None


class Output(BaseModel):
    priority: Literal[1, 2, 3, 4, 5]
    reasoning: str
    message: str
    topic: str | None = None
    urgency: str | None = None
    customer_id: str | None = None


def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("`").strip()
    return json.loads(text)


@yard.agent(
    name="ticket-priority-scorer",
    namespace="acme/support",
    intent="Score a support ticket's handling priority from 1-5 with reasoning",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="bedrock"),
    ],
    memory=MemoryContract(reads=[], writes=["priority_result"]),
    port=9110,
)
async def score(input: Input, ctx) -> Output:
    """Score a support ticket's priority using Bedrock."""
    message = input.message.strip()
    if not message:
        return Output(
            priority=1,
            reasoning="Empty message",
            message="",
            topic=input.topic,
            urgency=input.urgency,
            customer_id=input.customer_id,
        )

    prompt = SCORE_PROMPT.format(
        message=message[:2000],
        topic=input.topic or "unknown",
        urgency=input.urgency or "unknown",
    )
    response = await ctx.llm.complete(prompt)
    try:
        parsed = _parse_llm_json(response)
        priority = int(parsed.get("priority", 3))
        priority = min(5, max(1, priority))
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("score_parse_failed err=%s text=%s", e, str(response)[:200])
        parsed = {"reasoning": "Could not parse model output — defaulted to medium priority."}
        priority = 3

    result = Output(
        priority=priority,
        reasoning=parsed.get("reasoning", "No reasoning provided."),
        message=message,
        topic=input.topic,
        urgency=input.urgency,
        customer_id=input.customer_id,
    )
    ctx.memory["priority_result"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
