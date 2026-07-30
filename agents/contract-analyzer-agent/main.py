"""Contract Analyzer Agent — LLM-powered contract analysis.

v2 SDK: pure business logic. Extracts key terms, identifies risks, and
summarizes obligations from contract text.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentyard.v2 import MemoryContract, Resource, yard

SYSTEM_PROMPT = """You are an expert contract analyst specializing in financial services agreements.
Analyze the provided contract text and return a JSON object with this exact structure:

{
  "summary": "Brief 2-3 sentence summary of the contract",
  "key_terms": {
    "parties": ["Party A name", "Party B name"],
    "effective_date": "date string",
    "termination_clause": "summary of termination terms",
    "governing_law": "jurisdiction",
    "payment_terms": "payment details",
    "confidentiality": "NDA/confidentiality summary",
    "liability_cap": "liability limitation details"
  },
  "risks": [
    {"clause": "clause reference", "risk": "description", "severity": "high|medium|low", "recommendation": "suggested action"}
  ],
  "risk_score": 0-100,
  "risk_level": "low|medium|high",
  "obligations": [
    {"party": "who", "obligation": "what they must do", "deadline": "when"}
  ]
}

Be thorough but concise. Focus on material terms and genuine risks. Return ONLY valid JSON."""


class Input(BaseModel):
    text: str
    type: Literal["full", "risks", "terms", "summary"] = "full"


class Output(BaseModel):
    summary: str = ""
    key_terms: dict[str, Any] = Field(default_factory=dict)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    risk_score: int = 0
    risk_level: str = "unknown"
    obligations: list[dict[str, Any]] = Field(default_factory=list)
    analysis_type: str = "full"


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
    name="contract-analyzer-agent",
    namespace="acme/legal",
    intent="Analyze contracts to extract key terms, identify risks, and summarize obligations",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(reads=[], writes=["contract_analysis"]),
    port=9030,
)
async def analyze_contract(input: Input, ctx) -> Output:
    """Analyze contract text for terms, risks, and obligations."""
    if not input.text:
        return Output(analysis_type=input.type)

    user_prompt = f"Analyze this contract. Analysis type: {input.type}\n\n---\n{input.text}\n---"
    llm_text = await ctx.llm.complete(f"{SYSTEM_PROMPT}\n\n{user_prompt}")
    parsed = _parse_json_response(llm_text)

    result = Output(
        summary=parsed.get("summary", ""),
        key_terms=parsed.get("key_terms") or {},
        risks=parsed.get("risks") or [],
        risk_score=int(parsed.get("risk_score", 0) or 0),
        risk_level=parsed.get("risk_level", "unknown"),
        obligations=parsed.get("obligations") or [],
        analysis_type=input.type,
    )
    ctx.memory["contract_analysis"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
