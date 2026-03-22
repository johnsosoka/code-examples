"""Demo: Worker Check-In Monitor middleware.

A research agent is given a multi-step task. It tends to get absorbed in
research and forget to send status updates. The CheckInMiddleware detects
the silence and injects a <system> reminder after ``max_silent_turns`` turns
without a ``send_message`` call.

Usage:
    poetry run python -m bsm_examples.drift_detection.demo
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from bsm_examples.drift_detection.middleware import CheckInMiddleware  # noqa: E402
from bsm_examples.drift_detection.models import CheckInConfig  # noqa: E402
from bsm_examples.drift_detection.state import CheckInAgentState  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool
def send_message(message: str) -> str:
    """Send a status update message to the team."""
    return f"Message sent: {message}"


@tool
def do_research(topic: str) -> str:
    """Research a topic and return findings."""
    return (
        f"Research findings on '{topic}': [detailed findings here — "
        "see internal docs for full analysis]"
    )


@tool
def write_report(content: str) -> str:
    """Write a section of the report."""
    return f"Report section written: {content[:50]}..."


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a diligent research agent working on a multi-part project. "
    "Use the available tools to gather information and compile a report. "
    "Keep the team informed of your progress via the `send_message` tool. "
    "You will receive <system> tags with guidance during the task — follow them."
)


def build_agent() -> tuple[object, CheckInMiddleware]:
    """Build the agent and middleware. Returns both for state inspection."""
    config = CheckInConfig(
        target_tool="send_message",
        max_silent_turns=2,
        max_interventions=3,
        cooldown_turns=1,
    )
    middleware = CheckInMiddleware(config)
    agent = create_agent(
        model=ChatOpenAI(model="gpt-4.1-mini"),
        tools=[send_message, do_research, write_report],
        system_prompt=SYSTEM_PROMPT,
        middleware=[middleware],
        state_schema=CheckInAgentState,
    )
    return agent, middleware


def run_demo() -> None:
    """Run a 5-turn research conversation showing check-in monitoring in action."""
    print("=" * 60)
    print("WORKER CHECK-IN MONITOR DEMO")
    print("=" * 60)
    print("Monitoring: send_message (max silent turns: 2)")
    print()

    agent, middleware = build_agent()

    result = agent.invoke(  # type: ignore[attr-defined]
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Please research and report on the following topics: "
                        "1. Recent advances in renewable energy storage, "
                        "2. Economic impact of offshore wind farms, "
                        "3. Grid-scale battery technology. "
                        "This is a multi-step task — be thorough and keep us posted."
                    )
                )
            ]
        }
    )

    turns_since = result.get("turns_since_check_in", "N/A")
    interventions = result.get("intervention_count", "N/A")

    print()
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"  turns_since_check_in  : {turns_since}")
    print(f"  intervention_count    : {interventions}")
    print()
    print("  Conversation:")
    for msg in result["messages"]:
        role = type(msg).__name__
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"  [{role}] {content[:160]}")


def main() -> None:
    """Entry point for the demo."""
    run_demo()


if __name__ == "__main__":
    main()
