"""Invoice Processor Agent — extracts structured data from invoice text.

v2 SDK: pure business logic. Runtime handles transport and infrastructure.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from agentyard.v2 import MemoryContract, Resource, yard

SYSTEM_PROMPT = (
    "You are an accounts payable specialist. Extract structured invoice data from the text. "
    "Return ONLY valid JSON with fields: vendor, invoice_number, date, line_items (array of "
    "{description, quantity, unit_price, amount}), subtotal, tax, total, currency, payment_terms, po_number. "
    "Use null for fields you cannot determine."
)


class Input(BaseModel):
    text: str
    currency: str = "USD"


class Output(BaseModel):
    vendor: str = "Unknown"
    invoice_number: str = "Unknown"
    date: str = "Unknown"
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    currency: str = "USD"
    payment_terms: str = "Unknown"
    po_number: str | None = None
    confidence: float = 0.0


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
    name="invoice-processor-agent",
    namespace="acme/finance",
    intent="Extract structured data from invoice text (vendor, line items, tax, total)",
    inputs=Input,
    outputs=Output,
    is_idempotent=True,
    is_long_running=False,
    needs=[
        Resource.llm(provider="anthropic"),
        Resource.secrets(["ANTHROPIC_API_KEY"]),
    ],
    memory=MemoryContract(reads=[], writes=["invoice_extraction"]),
    port=9040,
)
async def invoice_processor(input: Input, ctx) -> Output:
    """Extract structured invoice fields from raw text."""
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Currency: {input.currency}\n\nInvoice text:\n{input.text}"
    )
    llm_text = await ctx.llm.complete(prompt)
    parsed = _parse_json(llm_text)

    result = Output(
        vendor=parsed.get("vendor") or "Unknown",
        invoice_number=parsed.get("invoice_number") or "Unknown",
        date=parsed.get("date") or "Unknown",
        line_items=parsed.get("line_items") or [],
        subtotal=float(parsed.get("subtotal") or 0.0),
        tax=float(parsed.get("tax") or 0.0),
        total=float(parsed.get("total") or 0.0),
        currency=parsed.get("currency") or input.currency,
        payment_terms=parsed.get("payment_terms") or "Unknown",
        po_number=parsed.get("po_number"),
        confidence=0.92 if parsed else 0.0,
    )
    ctx.memory["invoice_extraction"] = result.model_dump()
    return result


if __name__ == "__main__":
    yard.run()
