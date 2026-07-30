"""Fraud Detector Agent — analyzes transaction patterns for fraud indicators.

v2 SDK: pure business logic. Uses Claude to score risk against recent history.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentyard.v2 import MemoryContract, Resource, yard

SYSTEM_PROMPT = (
    "You are a fraud detection specialist at a major bank. Analyze the transaction against "
    "the customer's history for fraud indicators. Consider: velocity patterns, amount anomalies, "
    "geographic inconsistencies, time-of-day patterns, merchant risk. Return ONLY valid JSON with: "
    "risk_score (0-1), risk_level (low/medium/high), flags (array of strings), "
    "recommended_action (allow/review/block), reasoning (string)."
)


class Input(BaseModel):
    transaction: dict[str, Any]
    history: list[dict[str, Any]] = Field(default_factory=list)


class Output(BaseModel):
    risk_score: float = 0.0
    risk_level: Literal["low", "medium", "high"] = "low"
    flags: list[str] = Field(default_factory=list)
    recommended_action: Literal["allow", "review", "block"] = "allow"
    reasoning: str = ""


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
    name="fraud-detector-agent",
    namespace="acme/security",
    intent="Analyze transaction patterns to detect fraud indicators and recommend an action",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(reads=[], writes=["fraud_assessment"]),
    port=9041,
)
async def fraud_detector(input: Input, ctx) -> Output:
    """Score a transaction for fraud risk given recent history."""
    user_prompt = (
        f"Transaction:\n{json.dumps(input.transaction, indent=2)}\n\n"
        f"History ({len(input.history)} records):\n{json.dumps(input.history[-10:], indent=2)}"
    )
    llm_text = await ctx.llm.complete(f"{SYSTEM_PROMPT}\n\n{user_prompt}")
    parsed = _parse_json(llm_text)

    risk_score = float(parsed.get("risk_score") or 0.0)
    risk_level = parsed.get("risk_level")
    if risk_level not in ("low", "medium", "high"):
        risk_level = "low" if risk_score < 0.3 else ("medium" if risk_score < 0.6 else "high")
    action = parsed.get("recommended_action")
    if action not in ("allow", "review", "block"):
        action = "block" if risk_score >= 0.7 else ("review" if risk_score >= 0.4 else "allow")

    result = Output(
        risk_score=round(risk_score, 3),
        risk_level=risk_level,
        flags=parsed.get("flags") or [],
        recommended_action=action,
        reasoning=parsed.get("reasoning", ""),
    )
    ctx.memory["fraud_assessment"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
