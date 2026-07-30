"""ReAct-style reasoning loop.

Given a goal and a set of tools, the framework runs an LLM loop where
the model picks tools, sees their results, and keeps reasoning until it
either reaches the goal or hits the step cap. Agents don't have to
hand-write the loop — they call ``ctx.reason(goal, tools=[...])`` and
receive a ``ReasoningResult`` summarising the trace plus the final
answer.

Pairs naturally with :class:`agentyard.llm.LLMClient` (for the model
calls) and :meth:`agentyard.context.YardContext.tool` (for tool
invocations).

Usage:
    @yard.agent(name="researcher")
    async def research(input, ctx):
        result = await ctx.reason(
            goal=f"Answer concisely: {input['question']}",
            tools=["web_search", "calculator"],
            model="gpt-4o",
            max_steps=8,
        )
        return {"answer": result.answer, "tokens": result.total_tokens}
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agentyard.reasoning")


REACT_SYSTEM_PROMPT = """You are an autonomous agent that reasons step-by-step to achieve a goal.

At each step, decide whether to:
1. Call a tool to gather information
2. Produce the final answer

Available tools:
{tools_desc}

Format your response as JSON on a single line.

To call a tool:
{{"thought": "...", "action": {{"tool": "name", "arguments": {{...}}}}}}

To produce the answer:
{{"thought": "...", "final_answer": "..."}}

Be concise. Always output valid single-line JSON. Do not wrap the JSON in markdown."""


@dataclass
class ReasoningStep:
    index: int
    thought: str
    action: dict | None = None
    observation: Any = None
    error: str | None = None


@dataclass
class ReasoningResult:
    goal: str
    answer: str
    steps: list[ReasoningStep] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    finished: bool = False
    finish_reason: str = "completed"  # completed | max_steps | parse_error | tool_error

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "answer": self.answer,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "steps": [
                {
                    "index": s.index,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


class ReasoningError(Exception):
    pass


class Reasoner:
    """Implementation of ctx.reason() — the ReAct loop."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

    async def run(
        self,
        goal: str,
        *,
        tools: list[str] | None = None,
        tool_descriptions: dict[str, str] | None = None,
        model: str = "gpt-4o",
        max_steps: int = 6,
        temperature: float = 0.2,
        max_tokens_per_step: int = 512,
    ) -> ReasoningResult:
        """Run the reasoning loop until termination."""
        tools = tools or []
        result = ReasoningResult(goal=goal, answer="")

        tools_desc = self._describe_tools(tools, tool_descriptions or {})
        messages: list[dict] = [
            {
                "role": "system",
                "content": REACT_SYSTEM_PROMPT.format(tools_desc=tools_desc),
            },
            {"role": "user", "content": f"Goal: {goal}"},
        ]

        for step_idx in range(max_steps):
            try:
                from agentyard.llm import Message  # local import to avoid cycles

                llm_response = await self.ctx.llm.complete(
                    prompt=[
                        Message(role=m["role"], content=m["content"]) for m in messages
                    ],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens_per_step,
                    use_cache=False,  # reasoning steps must never reuse cached completions
                )
            except Exception as exc:
                raise ReasoningError(
                    f"LLM call failed at step {step_idx}: {exc}"
                ) from exc

            result.total_tokens += llm_response.tokens_in + llm_response.tokens_out
            result.total_cost_usd += llm_response.cost_usd

            try:
                parsed = self._parse(llm_response.text)
            except Exception as exc:
                step = ReasoningStep(
                    index=step_idx,
                    thought=llm_response.text[:200],
                    error=f"parse_error: {exc}",
                )
                result.steps.append(step)
                messages.append(
                    {"role": "assistant", "content": llm_response.text}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid single-line "
                            "JSON. Try again."
                        ),
                    }
                )
                continue

            thought = str(parsed.get("thought", ""))[:1000]

            if "final_answer" in parsed:
                result.steps.append(ReasoningStep(index=step_idx, thought=thought))
                result.answer = str(parsed["final_answer"])
                result.finished = True
                result.finish_reason = "completed"
                return result

            action = parsed.get("action") or {}
            tool_name = action.get("tool")
            arguments = action.get("arguments") or {}

            step = ReasoningStep(index=step_idx, thought=thought, action=action)

            if not tool_name:
                step.error = "no_tool_specified"
                result.steps.append(step)
                messages.append(
                    {"role": "assistant", "content": llm_response.text}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Pick a tool or return final_answer.",
                    }
                )
                continue

            if tools and tool_name not in tools:
                step.error = f"tool_not_allowed: {tool_name}"
                result.steps.append(step)
                messages.append(
                    {"role": "assistant", "content": llm_response.text}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool '{tool_name}' is not in the allowed list: "
                            f"{tools}"
                        ),
                    }
                )
                continue

            try:
                step.observation = await self.ctx.tool(tool_name, arguments)
            except Exception as exc:
                step.error = f"tool_error: {exc}"
                result.steps.append(step)
                messages.append(
                    {"role": "assistant", "content": llm_response.text}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool call failed: {exc}",
                    }
                )
                continue

            result.steps.append(step)
            messages.append({"role": "assistant", "content": llm_response.text})
            obs_str = json.dumps(step.observation, default=str)[:1500]
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {obs_str}",
                }
            )

        # Ran out of steps
        result.answer = "I could not reach a final answer within the step budget."
        result.finished = False
        result.finish_reason = "max_steps"
        return result

    def _describe_tools(
        self,
        tools: list[str],
        descriptions: dict[str, str],
    ) -> str:
        if not tools:
            return "(no tools available; answer from general knowledge)"
        lines = []
        for t in tools:
            desc = descriptions.get(t, "tool — see registry")
            lines.append(f"- {t}: {desc}")
        return "\n".join(lines)

    def _parse(self, text: str) -> dict:
        """Extract the first JSON object from messy LLM output."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("no JSON object in response")
        return json.loads(m.group(0))
