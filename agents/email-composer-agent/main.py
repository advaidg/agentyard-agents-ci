"""Email Composer Agent — LLM-powered professional email drafting.

v2 SDK: pure business logic. Composes polished emails from purpose, tone,
context, and key points.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from agentyard.v2 import MemoryContract, Resource, yard

SYSTEM_PROMPT = """You are a professional email writing expert. Compose polished, effective emails.

Return a JSON object with this exact structure:
{
  "subject": "Email subject line",
  "body": "Full email body with proper greeting, paragraphs, and sign-off",
  "suggested_attachments": ["filename.ext"],
  "tone": "the tone used",
  "purpose": "the purpose",
  "word_count": 0,
  "tips": ["optional writing tips or notes about the email"]
}

Guidelines:
- Match the requested tone precisely (formal/friendly/urgent)
- Include a proper greeting with recipient name if provided
- Structure the body with clear paragraphs
- Include a professional sign-off
- Keep it concise but complete
- Incorporate all key points naturally
Return ONLY valid JSON."""


class Input(BaseModel):
    purpose: Literal["follow_up", "proposal", "complaint", "thank_you"]
    context: str = ""
    tone: Literal["formal", "friendly", "urgent"] = "formal"
    recipient: str = ""
    key_points: list[str] = Field(default_factory=list)


class Output(BaseModel):
    subject: str = ""
    body: str = ""
    suggested_attachments: list[str] = Field(default_factory=list)
    tone: str = "formal"
    purpose: str = "follow_up"
    word_count: int = 0
    tips: list[str] = Field(default_factory=list)


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
    name="email-composer-agent",
    namespace="acme/comms",
    intent="Compose professional emails based on purpose, tone, context, and key points",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(reads=[], writes=["composed_email"]),
    port=9033,
)
async def compose_email(input: Input, ctx) -> Output:
    """Draft a polished email matching the requested tone and purpose."""
    key_points_block = (
        "\n".join(f"- {p}" for p in input.key_points)
        if input.key_points
        else "None specified"
    )
    user_prompt = (
        f"Compose a {input.tone} {input.purpose} email.\n"
        f"Recipient: {input.recipient or 'Not specified'}\n"
        f"Context: {input.context or 'Not specified'}\n"
        f"Key points to include:\n{key_points_block}"
    )
    llm_text = await ctx.llm.complete(f"{SYSTEM_PROMPT}\n\n{user_prompt}")
    parsed = _parse_json_response(llm_text)

    result = Output(
        subject=parsed.get("subject", ""),
        body=parsed.get("body", ""),
        suggested_attachments=parsed.get("suggested_attachments") or [],
        tone=parsed.get("tone") or input.tone,
        purpose=parsed.get("purpose") or input.purpose,
        word_count=int(parsed.get("word_count", 0) or 0),
        tips=parsed.get("tips") or [],
    )
    ctx.memory["composed_email"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
