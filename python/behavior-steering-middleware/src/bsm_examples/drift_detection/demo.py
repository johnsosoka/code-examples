"""Demonstration of Behavior Drift Detection middleware with a real LLM.

Runs a 5-turn math conversation using a real OpenAI model. The middleware
monitors tool usage and injects corrective reminders when drift is detected.

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

from bsm_examples.drift_detection.middleware import (  # noqa: E402
    DriftDetectionConfig,
    DriftSteeringMiddleware,
    ToolUsagePattern,
)
from bsm_examples.drift_detection.state import DriftAgentState  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@tool
def calculate(a: float, b: float, operation: str) -> str:
    """Perform arithmetic. Operations: add, subtract, multiply, divide."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    result = ops.get(operation, 0)
    return f"{a} {operation} {b} = {result}"


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units. Supports miles/km, feet/meters."""
    conversions = {
        ("miles", "km"): 1.60934,
        ("km", "miles"): 0.621371,
        ("feet", "meters"): 0.3048,
        ("meters", "feet"): 3.28084,
    }
    factor = conversions.get((from_unit, to_unit), 1.0)
    return f"{value} {from_unit} = {value * factor:.2f} {to_unit}"


def build_agent() -> tuple[object, DriftSteeringMiddleware]:
    """Build the agent and middleware. Returns both for state inspection."""
    pattern = ToolUsagePattern(
        name="calculator_agent",
        expected_tools={
            "calculate": 0.8,
            "unit_convert": 0.2,
        },
        required_tools=["calculate"],
        minimum_frequencies={"calculate": 0.5},
    )
    config = DriftDetectionConfig(
        history_window_size=5,
        recency_bias=1.5,
        warning_threshold=0.3,
        intervention_threshold=0.6,
        max_interventions_per_conversation=3,
        cooldown_turns=1,
        include_statistics=True,
    )
    middleware = DriftSteeringMiddleware(pattern, config)
    agent = create_agent(
        model=ChatOpenAI(model="gpt-4.1-mini"),
        tools=[calculate, unit_convert],
        system_prompt=(
            "You are a calculator assistant. "
            "Always use the calculate or unit_convert tools for math. "
            "Never compute answers in your head — always call a tool. "
            "You will receive <system> tags with behavioral guidance "
            "— follow them."
        ),
        middleware=[middleware],
        state_schema=DriftAgentState,
    )
    return agent, middleware


def run_demo() -> None:
    """Run a 5-turn math conversation showing drift detection in action."""
    print("=" * 60)
    print("BEHAVIOR DRIFT DETECTION DEMO")
    print("=" * 60)
    print("Pattern: calculate (80%), unit_convert (20%)")
    print("Intervention threshold: 0.6")
    print()

    agent, middleware = build_agent()

    turns = [
        "What is 15 multiplied by 23?",
        "Convert 5 miles to kilometers.",
        "What is 144 divided by 12?",
        "What is 99 plus 1?",
        "What is 50 multiplied by 2?",
    ]

    messages: list[HumanMessage] = []

    for i, question in enumerate(turns, 1):
        print(f"--- Turn {i} ---")
        print(f"User: {question}")

        messages.append(HumanMessage(content=question))
        result = agent.invoke({"messages": messages})  # type: ignore[attr-defined]

        # Pull observability fields from agent state
        drift_score = result.get("drift_score", 0.0)
        intervention_count = result.get("intervention_count", 0)

        # Show the last AI message as the response
        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content[:120]}")
        print(f"  drift_score={drift_score:.3f}  interventions={intervention_count}")
        print()

        # Carry full message history forward for the next turn
        messages = list(result["messages"])

    # Final report from the middleware's internal detector
    report = middleware.get_drift_report()
    print("=" * 60)
    print("FINAL DRIFT REPORT")
    print("=" * 60)
    print(f"  Total turns recorded : {report['turn_count']}")
    print(f"  Interventions        : {report['intervention_count']}")
    print(f"  Final drift score    : {report['drift_score']:.3f}")
    print()
    print("  Tool usage (expected vs observed):")
    for tool_name, data in report["tool_comparison"].items():
        print(
            f"    {tool_name}: expected={data['expected']:.2f} "
            f"observed={data['observed']:.2f}"
        )
    print()


def main() -> None:
    """Entry point for the demo."""
    run_demo()


if __name__ == "__main__":
    main()
