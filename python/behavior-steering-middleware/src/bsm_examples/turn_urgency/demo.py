"""Demonstration of Turn Urgency middleware with a Shopping Recommendation Agent.

Runs a 5-turn product recommendation conversation. The middleware injects
progressively stricter urgency instructions as turns are consumed, ultimately
forcing the agent to call `recommend_product` on the final turn.

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
    UrgencyAgentState,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@tool
def search_products(query: str, category: str = "all") -> str:
    """Search for products matching a query in the catalog."""
    return (
        f"Found products for '{query}' in {category}: "
        "[SoundPro X1 ($79), AudioMax 500 ($149), BeatWave Pro ($229)]"
    )


@tool
def get_product_details(product_id: str) -> str:
    """Get detailed information about a specific product."""
    details = {
        "soundpro-x1": (
            "SoundPro X1 ($79): 20hr battery, basic noise cancellation, "
            "Bluetooth 5.0. Best for: casual listeners on a budget."
        ),
        "audiomax-500": (
            "AudioMax 500 ($149): 30hr battery, active noise cancellation, "
            "Bluetooth 5.2, foldable. Best for: commuters and office use."
        ),
        "beatwave-pro": (
            "BeatWave Pro ($229): 40hr battery, premium ANC, hi-fi audio, "
            "multipoint pairing. Best for: audiophiles and frequent travelers."
        ),
    }
    return details.get(product_id, f"Product '{product_id}' not found in catalog.")


@tool
def recommend_product(product_name: str, reason: str) -> str:
    """Recommend a product to the customer with a reason."""
    return f"Recommended '{product_name}': {reason}"


def build_agent() -> object:
    """Build the shopping agent with TurnUrgencyMiddleware configured for 5 turns."""
    config = TurnUrgencyConfig(
        max_turns=5,
        required_completion_tool="recommend_product",
    )
    middleware = TurnUrgencyMiddleware(config)
    agent = create_agent(
        model=ChatOpenAI(model="gpt-4.1-mini"),
        tools=[search_products, get_product_details, recommend_product],
        system_prompt=(
            "You are a helpful shopping assistant specializing in consumer electronics. "
            "Help customers find the right product for their needs. "
            "Use search_products to explore options, get_product_details for specifics, "
            "and recommend_product when you are ready to make a suggestion. "
            "You will receive <system> tags with behavioral guidance — follow them."
        ),
        middleware=[middleware],
        state_schema=UrgencyAgentState,
    )
    return agent


def run_demo() -> None:
    """Run a 5-turn shopping session showing urgency escalation toward a recommendation."""
    print("=" * 70)
    print("TURN URGENCY MIDDLEWARE DEMO — Shopping Recommendation Agent")
    print("Task: Help customer find wireless headphones — 5-turn budget")
    print("=" * 70)
    print()

    agent = build_agent()

    turns = [
        "Hi! I'm looking for wireless headphones. What do you have?",
        "I mostly use them for commuting and working from home. What's my best option?",
        "Can you tell me more about the AudioMax 500?",
        "That sounds good. My budget is around $150. Does it fit?",
        "Okay, I think I've heard enough. What do you recommend?",
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
        completion_tool_called = result.get("completion_tool_called", False)

        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content[:300]}")
        print(
            f"  urgency={urgency_level}  "
            f"turns={turns_used}/{5}  remaining={turns_remaining}  "
            f"recommend_called={completion_tool_called}"
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
