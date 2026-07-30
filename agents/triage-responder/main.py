"""Triage Responder — Step 3 of the customer support pipeline.

v2 SDK: drafts an empathetic reply based on the classified + scored input.
Pure business logic — no transport, no retry, no infrastructure.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel

from agentyard.v2 import MemoryContract, Resource, yard

logger = logging.getLogger("triage-responder")

REPLY_PROMPT = """You are a senior customer support agent at Acme Corp.

Draft an empathetic, professional reply to the customer message below. Keep it concise (3-5 sentences). Match the tone to the customer's emotional state.

Context from earlier triage steps:
- Topic: {topic}
- Urgency: {urgency}
- Customer sentiment: {sentiment} (frustration {frustration_score}/10)
- Churn risk: {churn_risk}
- Customer language: {language}

Customer message:
\"\"\"
{message}
\"\"\"

Respond with a JSON object containing:
- "reply": the drafted reply text in the customer's language
- "next_action": one of [send_as_is, route_to_human, escalate_to_manager, request_more_info]
- "suggested_tags": array of up to 3 short tags for the ticket
- "internal_note": one sentence for the human agent's eyes only

Respond with ONLY the JSON object, no markdown."""


class Input(BaseModel):
    message: str
    topic: str | None = None
    urgency: str | None = None
    summary: str | None = None
    sentiment: str | None = None
    frustration_score: int | None = None
    churn_risk: str | None = None
    language: str | None = None
    customer_id: str | None = None


class Output(BaseModel):
    reply: str
    next_action: Literal["send_as_is", "route_to_human", "escalate_to_manager", "request_more_info"]
    suggested_tags: list[str]
    internal_note: str
    topic: str | None = None
    urgency: str | None = None
    sentiment: str | None = None
    frustration_score: int | None = None
    summary: str | None = None
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
    name="triage-responder",
    namespace="acme/support",
    intent="Draft an empathetic reply tailored to the customer's topic and sentiment",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(
        reads=["classification_result", "sentiment_result"],
        writes=["draft_reply"],
    ),
    port=9103,
)
async def respond(input: Input, ctx) -> Output:
    """Draft a reply based on the classified + scored input."""
    message = input.message.strip()
    topic = input.topic or "general"
    if not message:
        return Output(
            reply="We received an empty message — please share details so we can help.",
            next_action="request_more_info",
            suggested_tags=["empty"],
            internal_note="Empty message received",
            topic=topic,
            urgency=input.urgency,
            sentiment=input.sentiment,
            frustration_score=input.frustration_score,
            summary=input.summary,
            customer_id=input.customer_id,
        )

    prompt = REPLY_PROMPT.format(
        topic=topic,
        urgency=input.urgency or "medium",
        sentiment=input.sentiment or "neutral",
        frustration_score=input.frustration_score or 5,
        churn_risk=input.churn_risk or "medium",
        language=input.language or "en",
        message=message[:2000],
    )
    response = await ctx.llm.complete(prompt)
    try:
        parsed = _parse_llm_json(response)
    except json.JSONDecodeError as e:
        logger.warning("responder_parse_failed err=%s text=%s", e, str(response)[:200])
        parsed = {
            "reply": "Thanks for reaching out — a member of our team will be in touch shortly.",
            "next_action": "route_to_human",
            "suggested_tags": [topic],
            "internal_note": "LLM response did not parse; routed to human",
        }

    result = Output(
        reply=parsed.get("reply", ""),
        next_action=parsed.get("next_action", "route_to_human"),
        suggested_tags=parsed.get("suggested_tags") or [],
        internal_note=parsed.get("internal_note", ""),
        topic=topic,
        urgency=input.urgency,
        sentiment=input.sentiment,
        frustration_score=input.frustration_score,
        summary=input.summary,
        customer_id=input.customer_id,
    )
    ctx.memory["draft_reply"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
