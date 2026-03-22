"""Demonstration of Turn Urgency middleware with a real LLM.

Runs a 5-turn debugging conversation. The middleware injects progressively
stricter urgency instructions into the system prompt as turns are consumed.

Requires OPENAI_API_KEY in the environment or a .env file.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from bsm_examples.turn_urgency import (  # noqa: E402
    TurnUrgencyConfig,
    TurnUrgencyMiddleware,
)
from bsm_examples.turn_urgency.types import UrgencyAgentState  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@tool
def analyze_code(code_snippet: str) -> str:
    """Analyze a code snippet for common issues like type errors, null references, etc."""
    issues = []
    if "None" in code_snippet or "null" in code_snippet:
        issues.append("Potential null reference detected")
    if ".process()" in code_snippet:
        issues.append("Method call on potentially uninitialized object")
    return (
        f"Found {len(issues)} issue(s): {'; '.join(issues)}"
        if issues
        else "No issues detected"
    )


def build_agent() -> object:
    """Build the agent with TurnUrgencyMiddleware configured for 5 turns."""
    config = TurnUrgencyConfig(max_turns=5)
    middleware = TurnUrgencyMiddleware(config)
    agent = create_agent(
        model=ChatOpenAI(model="gpt-4.1-mini"),
        tools=[analyze_code],
        system_prompt=(
            "You are a Python debugging assistant. "
            "Use the analyze_code tool when the user shares code. "
            "You will receive <system> tags with behavioral guidance "
            "— follow them."
        ),
        middleware=[middleware],
        state_schema=UrgencyAgentState,
    )
    return agent


def run_demo() -> None:
    """Run a 5-turn debugging session showing urgency escalation."""
    print("=" * 70)
    print("TURN URGENCY MIDDLEWARE DEMO")
    print("Task: Debug a Python TypeError — 5-turn budget")
    print("=" * 70)
    print()

    agent = build_agent()

    turns = [
        "My Python script crashes with a TypeError. How do I debug it?",
        "The error is on line 42: 'NoneType' object has no attribute 'process'.",
        "Here is the code: result = data.process()",
        "I added the None check and that error is gone, but now there's an AttributeError.",
        "What about adding logging so I can trace the issue next time?",
    ]

    messages: list[HumanMessage] = []

    for i, question in enumerate(turns, 1):
        print(f"--- Turn {i}/5 ---")
        print(f"User: {question}")

        messages.append(HumanMessage(content=question))
        result = agent.invoke({"messages": messages})  # type: ignore[attr-defined]

        urgency_level = result.get("urgency_level", "UNKNOWN")
        turns_used = result.get("turns_used", i)
        turns_remaining = result.get("turns_remaining", 5 - i)

        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content[:200]}")
        print(
            f"  urgency_level={urgency_level}  "
            f"turns_used={turns_used}  turns_remaining={turns_remaining}"
        )
        print()

        messages = list(result["messages"])

    print("=" * 70)
    print("DEMO COMPLETE")
    print("Urgency progression: EXPLORATION -> BALANCED -> EFFICIENCY -> TERMINAL")
    print("=" * 70)
    print()


def main() -> None:
    """Entry point for the demo."""
    run_demo()


if __name__ == "__main__":
    main()
