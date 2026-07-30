"""Triage Sentiment Analyzer — Step 2 of the customer support pipeline.

v2 SDK: pure business logic. Adds sentiment, frustration, and churn risk
signals to the payload coming from the classifier.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel

from agentyard.v2 import MemoryContract, Resource, yard

logger = logging.getLogger("triage-sentiment")

SENTIMENT_PROMPT = """You are a customer support sentiment analyst.

Analyze the customer message below and respond with a JSON object containing:
- "sentiment": one of [positive, neutral, mildly_negative, frustrated, angry]
- "frustration_score": integer 0-10 (0 = happy, 10 = furious)
- "churn_risk": one of [low, medium, high]
- "key_phrases": array of up to 3 phrases that drove the assessment

Customer message:
\"\"\"
{message}
\"\"\"

Respond with ONLY the JSON object, no markdown, no commentary."""


class Input(BaseModel):
    message: str
    topic: str | None = None
    urgency: str | None = None
    summary: str | None = None
    language: str | None = None
    customer_id: str | None = None


class Output(BaseModel):
    topic: str | None = None
    urgency: str
    summary: str | None = None
    language: str | None = None
    sentiment: Literal["positive", "neutral", "mildly_negative", "frustrated", "angry"]
    frustration_score: int
    churn_risk: Literal["low", "medium", "high"]
    key_phrases: list[str]
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
    name="triage-sentiment",
    namespace="acme/support",
    intent="Analyze sentiment and churn risk for customer support messages",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(
        reads=["classification_result"],
        writes=["sentiment_result"],
    ),
    port=9102,
)
async def analyze(input: Input, ctx) -> Output:
    """Analyze sentiment for a classified customer message."""
    message = input.message.strip()
    urgency = input.urgency or "medium"
    if not message:
        return Output(
            topic=input.topic,
            urgency=urgency,
            summary=input.summary,
            language=input.language,
            sentiment="neutral",
            frustration_score=0,
            churn_risk="low",
            key_phrases=[],
            message="",
            customer_id=input.customer_id,
        )

    response = await ctx.llm.complete(SENTIMENT_PROMPT.format(message=message[:2000]))
    try:
        parsed = _parse_llm_json(response)
    except json.JSONDecodeError as e:
        logger.warning("sentiment_parse_failed err=%s text=%s", e, str(response)[:200])
        parsed = {
            "sentiment": "neutral",
            "frustration_score": 5,
            "churn_risk": "medium",
            "key_phrases": [],
        }

    score = int(parsed.get("frustration_score", 5) or 5)
    # Auto-bump urgency if sentiment is severe
    if score >= 8 and urgency in ("low", "medium"):
        urgency = "high"

    result = Output(
        topic=input.topic,
        urgency=urgency,
        summary=input.summary,
        language=input.language,
        sentiment=parsed.get("sentiment", "neutral"),
        frustration_score=score,
        churn_risk=parsed.get("churn_risk", "medium"),
        key_phrases=parsed.get("key_phrases") or [],
        message=message,
        customer_id=input.customer_id,
    )
    ctx.memory["sentiment_result"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
