"""KYC Reviewer Agent — LLM-powered KYC/AML compliance review.

v2 SDK: pure business logic. Produces a risk scored KYC assessment with flags
and recommended next steps.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentyard.v2 import MemoryContract, Resource, yard

SYSTEM_PROMPT = """You are a KYC/AML compliance expert. Review the customer information and perform a thorough KYC assessment.

Return a JSON object with this exact structure:
{
  "customer_name": "name",
  "risk_level": "low|medium|high",
  "risk_score": 0.0-1.0,
  "flags": [
    {"type": "flag_category", "detail": "description", "severity": "high|medium|low"}
  ],
  "recommendations": ["action item 1", "action item 2"],
  "confidence": 0.0-1.0,
  "checks_performed": ["check1", "check2"],
  "detailed_assessment": "Narrative assessment paragraph"
}

Check for: sanctions list matches, PEP indicators, high-risk jurisdictions, suspicious source of funds,
incomplete documentation, unusual patterns, adverse media indicators.
Return ONLY valid JSON."""


class Input(BaseModel):
    customer_name: str
    customer_data: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    customer_name: str
    risk_level: Literal["low", "medium", "high"] = "low"
    risk_score: float = 0.0
    flags: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    checks_performed: list[str] = Field(default_factory=list)
    detailed_assessment: str = ""


def _parse_json_response(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


@yard.agent(
    name="kyc-reviewer-agent",
    namespace="acme/compliance",
    intent="Perform KYC/AML compliance review with risk scoring and recommendations",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(reads=[], writes=["kyc_review"]),
    port=9031,
)
async def review_kyc(input: Input, ctx) -> Output:
    """Assess a customer against KYC/AML criteria."""
    user_prompt = (
        f"Perform KYC review for customer:\n"
        f"Name: {input.customer_name}\n"
        f"Customer Data: {json.dumps(input.customer_data, indent=2)}\n"
    )
    llm_text = await ctx.llm.complete(f"{SYSTEM_PROMPT}\n\n{user_prompt}")
    parsed = _parse_json_response(llm_text)

    risk_level = parsed.get("risk_level")
    if risk_level not in ("low", "medium", "high"):
        risk_level = "low"

    result = Output(
        customer_name=parsed.get("customer_name") or input.customer_name,
        risk_level=risk_level,
        risk_score=float(parsed.get("risk_score") or 0.0),
        flags=parsed.get("flags") or [],
        recommendations=parsed.get("recommendations") or [],
        confidence=float(parsed.get("confidence") or 0.0),
        checks_performed=parsed.get("checks_performed") or [],
        detailed_assessment=parsed.get("detailed_assessment", ""),
    )
    ctx.memory["kyc_review"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
