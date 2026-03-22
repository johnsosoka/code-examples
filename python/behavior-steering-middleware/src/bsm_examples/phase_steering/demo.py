"""Demonstration of Tool-Triggered Phase Steering middleware with a real LLM.

Simulates a personalization shopping assistant that progresses through
DISCOVERY -> CURATION -> RECOMMENDATION -> COMPLETE phases. Each phase
transition is triggered by a specific tool call, not heuristics.

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

from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware  # noqa: E402
from bsm_examples.phase_steering.models import (  # noqa: E402
    Phase,
    PhaseBehavior,
    PhaseConfig,
    PhaseTransitionRule,
)
from bsm_examples.phase_steering.state import PhaseAgentState  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool
def browse_catalog(category: str) -> str:
    """Browse products in a catalog category."""
    return f"Catalog for '{category}': [list of products with IDs and prices]"


@tool
def note_preferences(preferences: str) -> str:
    """Record the customer's stated preferences and requirements."""
    return f"Preferences noted: {preferences}"


@tool
def filter_products(criteria: str) -> str:
    """Filter products based on specific criteria."""
    return f"Filtered products matching '{criteria}': [filtered results]"


@tool
def compare_options(items: str) -> str:
    """Compare multiple products side by side."""
    item_list = [i.strip() for i in items.split(",")]
    return f"Comparison of {len(item_list)} items: [detailed comparison]"


@tool
def shortlist_items(items: str) -> str:
    """Create a shortlist of top candidate products."""
    return f"Shortlisted: {items}"


@tool
def present_recommendation(recommendation: str) -> str:
    """Present the final product recommendation to the customer."""
    return f"Final recommendation: {recommendation}"


# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------


def create_personalization_config() -> PhaseConfig:
    """Build the phase configuration for a personalization shopping workflow.

    Transitions are tool-triggered:
    - DISCOVERY  -> CURATION:     ``note_preferences``
    - CURATION   -> RECOMMENDATION: ``shortlist_items``
    - RECOMMENDATION -> COMPLETE: ``present_recommendation``
    """
    return PhaseConfig(
        phases={
            Phase.DISCOVERY: PhaseBehavior(
                system_prompt_addition=(
                    "Ask questions to understand what the customer is looking for. "
                    "Browse the catalog to explore options. Once you have a good "
                    "understanding of their needs, use the note_preferences tool."
                ),
                allowed_tools=["browse_catalog", "note_preferences"],
                max_iterations=5,
            ),
            Phase.CURATION: PhaseBehavior(
                system_prompt_addition=(
                    "Filter and compare products based on the customer's preferences. "
                    "When you have identified the best options, use shortlist_items "
                    "to narrow down."
                ),
                allowed_tools=["filter_products", "compare_options", "shortlist_items"],
                max_iterations=5,
            ),
            Phase.RECOMMENDATION: PhaseBehavior(
                system_prompt_addition=(
                    "Present your final recommendation to the customer with clear reasoning."
                ),
                allowed_tools=["present_recommendation"],
                max_iterations=2,
            ),
            Phase.COMPLETE: PhaseBehavior(
                system_prompt_addition="Workflow complete.",
                allowed_tools=[],
                max_iterations=1,
            ),
        },
        transitions=[
            PhaseTransitionRule(
                from_phase=Phase.DISCOVERY,
                to_phase=Phase.CURATION,
                trigger_tool="note_preferences",
                description="Customer preferences captured; move to curation",
            ),
            PhaseTransitionRule(
                from_phase=Phase.CURATION,
                to_phase=Phase.RECOMMENDATION,
                trigger_tool="shortlist_items",
                description="Top candidates identified; move to recommendation",
            ),
            PhaseTransitionRule(
                from_phase=Phase.RECOMMENDATION,
                to_phase=Phase.COMPLETE,
                trigger_tool="present_recommendation",
                description="Recommendation delivered; workflow complete",
            ),
        ],
        initial_phase=Phase.DISCOVERY,
    )


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def run_demo() -> None:
    """Run a multi-turn personalization session walking through all phases."""
    print("=" * 64)
    print("TOOL-TRIGGERED PHASE STEERING DEMO: Personalization Assistant")
    print("=" * 64)
    print()

    config = create_personalization_config()
    middleware = PhaseSteeringMiddleware(config)

    all_tools = [
        browse_catalog,
        note_preferences,
        filter_products,
        compare_options,
        shortlist_items,
        present_recommendation,
    ]

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4.1-mini"),
        tools=all_tools,
        system_prompt=(
            "You are a personalization shopping assistant. "
            "You will receive <system> tags with phase guidance — follow them carefully "
            "and use only the tools specified for the current phase."
        ),
        middleware=[middleware],
        state_schema=PhaseAgentState,
    )

    conversation: list[HumanMessage] = []

    turns = [
        "Hi! I'm looking for a new laptop. I work in software development and do some gaming.",
        "I'd like something under $2000, prefer 16GB RAM minimum, and good battery life.",
        "Can you narrow it down to the top 2-3 options and make a recommendation?",
    ]

    for i, user_message in enumerate(turns, 1):
        print(f"--- Turn {i} ---")
        print(f"User: {user_message}")

        conversation.append(HumanMessage(content=user_message))
        result = agent.invoke({"messages": conversation})  # type: ignore[call-overload]

        current_phase = result.get("current_phase", "UNKNOWN")
        phase_iteration = result.get("phase_iteration", 0)
        total_iterations = result.get("total_iterations", 0)

        last_message = result["messages"][-1]
        response_preview = str(last_message.content)[:300]
        print(f"Agent: {response_preview}")
        print(
            f"  [phase={current_phase} | phase_iter={phase_iteration} | total={total_iterations}]"
        )
        print()

        conversation = list(result["messages"])

    print("=" * 64)
    print("DEMO COMPLETE")
    print(f"Final phase: {middleware.current_phase.name}")
    transitions = [(p.name, itr) for p, itr in middleware.phase_history]
    print(f"Phase transitions: {transitions}")
    print("=" * 64)
    print()


def main() -> None:
    """Entry point for the demo."""
    run_demo()


if __name__ == "__main__":
    main()
