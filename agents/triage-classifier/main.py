"""Triage Classifier — Step 1 of the customer support pipeline.

v2 SDK: pure business logic only. Transport, retries, and infrastructure are
handled by the runtime.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel

from agentyard.v2 import MemoryContract, Resource, yard

logger = logging.getLogger("triage-classifier")

CLASSIFY_PROMPT = """You are a customer support triage classifier.

Read the customer message below and respond with a JSON object containing:
- "topic": one of [billing, technical, account, shipping, product, general]
- "urgency": one of [low, medium, high, critical]
- "summary": a one-sentence summary of the issue (max 20 words)
- "language": ISO 639-1 code of the customer's language (e.g. "en", "es")

Customer message:
\"\"\"
{message}
\"\"\"

Respond with ONLY the JSON object, no markdown, no commentary."""


class Input(BaseModel):
    message: str
    customer_id: str | None = None


class Output(BaseModel):
    topic: Literal["billing", "technical", "account", "shipping", "product", "general"]
    urgency: Literal["low", "medium", "high", "critical"]
    summary: str
    language: str
    message: str
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
    name="triage-classifier",
    namespace="acme/support",
    intent="Classify customer support messages by topic and urgency",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(reads=[], writes=["classification_result"]),
    port=9101,
)
async def classify(input: Input, ctx) -> Output:
    """Classify a customer support message by topic and urgency."""
    message = input.message.strip()
    if not message:
        return Output(
            topic="general",
            urgency="low",
            summary="Empty message",
            language="en",
            message="",
            customer_id=input.customer_id,
        )

    response = await ctx.llm.complete(CLASSIFY_PROMPT.format(message=message[:2000]))
    try:
        parsed = _parse_llm_json(response)
    except json.JSONDecodeError as e:
        logger.warning("classify_parse_failed err=%s text=%s", e, str(response)[:200])
        parsed = {
            "topic": "general",
            "urgency": "medium",
            "summary": message[:80],
            "language": "en",
        }

    result = Output(
        topic=parsed.get("topic", "general"),
        urgency=parsed.get("urgency", "medium"),
        summary=parsed.get("summary", message[:80]),
        language=parsed.get("language", "en"),
        message=message,
        customer_id=input.customer_id,
    )
    ctx.memory["classification_result"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
