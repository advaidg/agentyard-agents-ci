"""Report Generator Agent — LLM-powered business report generation.

v2 SDK: pure business logic. Produces structured reports (financial,
compliance, performance, summary) from input data.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentyard.v2 import MemoryContract, Resource, yard

SYSTEM_PROMPT = """You are a business report generation expert. Create comprehensive, well-structured reports from data.

Return a JSON object with this exact structure:
{
  "title": "Report title",
  "sections": [
    {"heading": "Section heading", "content": "Section content with analysis"}
  ],
  "executive_summary": "2-3 sentence executive summary",
  "key_findings": ["Finding 1", "Finding 2"],
  "report_type": "type",
  "format": "narrative|bullets|executive_summary",
  "recommendations": ["Recommendation 1"],
  "data_quality_notes": "Any observations about data completeness"
}

Guidelines:
- For 'narrative' format: Write flowing paragraphs with analysis
- For 'bullets' format: Use bullet-point lists for each section
- For 'executive_summary' format: Be extremely concise, max 5 bullet points total
- Always include actionable insights, not just data recitation
- Identify trends and anomalies in the data
Return ONLY valid JSON."""


class Input(BaseModel):
    data: dict[str, Any]
    report_type: Literal["financial", "compliance", "performance", "summary"] = "summary"
    format: Literal["narrative", "bullets", "executive_summary"] = "narrative"


class Output(BaseModel):
    title: str = ""
    sections: list[dict[str, Any]] = Field(default_factory=list)
    executive_summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    report_type: str = "summary"
    format: str = "narrative"
    recommendations: list[str] = Field(default_factory=list)
    data_quality_notes: str = ""


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
    name="report-generator-agent",
    namespace="acme/analytics",
    intent="Generate structured business reports (financial, compliance, performance, summary)",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(reads=[], writes=["generated_report"]),
    port=9035,
)
async def generate_report(input: Input, ctx) -> Output:
    """Generate a structured report from input data."""
    if not input.data:
        return Output(report_type=input.report_type, format=input.format)

    user_prompt = (
        f"Generate a {input.report_type} report in {input.format} format from this data:\n\n"
        f"{json.dumps(input.data, indent=2)}"
    )
    llm_text = await ctx.llm.complete(f"{SYSTEM_PROMPT}\n\n{user_prompt}")
    parsed = _parse_json_response(llm_text)

    result = Output(
        title=parsed.get("title", ""),
        sections=parsed.get("sections") or [],
        executive_summary=parsed.get("executive_summary", ""),
        key_findings=parsed.get("key_findings") or [],
        report_type=parsed.get("report_type") or input.report_type,
        format=parsed.get("format") or input.format,
        recommendations=parsed.get("recommendations") or [],
        data_quality_notes=parsed.get("data_quality_notes", ""),
    )
    ctx.memory["generated_report"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
