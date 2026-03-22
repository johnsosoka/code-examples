"""Demonstration of Tool-Triggered Phase Steering middleware with a real LLM.

Simulates a personalization shopping assistant that progresses through
DISCOVERY -> CURATION -> RECOMMENDATION -> COMPLETE phases. Each phase
transition is triggered by a specific tool call, not heuristics.

Requires OPENAI_API_KEY in the environment or a .env file.
"""

from __future__ import annotations

import logging
from typing import Any

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


def _print_turn(
    turn: int,
    label: str,
    user_msg: str,
    result: dict[str, Any],
    middleware: PhaseSteeringMiddleware,
) -> None:
    """Print a single turn's results."""
    phase = result.get("current_phase", "UNKNOWN")
    phase_iter = result.get("phase_iteration", 0)
    total = result.get("total_iterations", 0)
    last = result["messages"][-1]
    preview = str(last.content)[:300]

    print(f"--- Turn {turn}: {label} ---")
    print(f"User: {user_msg}")
    print(f"Agent: {preview}")
    print(f"  [phase={phase} | iter={phase_iter} | total={total}]")
    print()


def run_demo() -> None:
    """Run a multi-turn personalization session, one phase per turn.

    Each turn is scoped to a single phase. The user drives the workflow
    forward — the agent only uses tools available in the current phase,
    and phase transitions fire when the user explicitly asks the agent
    to use the trigger tool.
    """
    print("=" * 64)
    print("TOOL-TRIGGERED PHASE STEERING DEMO")
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
            "You will receive <framework_instruction> tags with phase "
            "guidance — follow them and use only the tools specified."
        ),
        middleware=[middleware],
        state_schema=PhaseAgentState,
    )

    conversation: list[HumanMessage] = []

    # Each turn is scoped to one phase. The user drives transitions by
    # asking the agent to perform the trigger action in a follow-up turn.
    turns = [
        # Turn 1 — DISCOVERY: browse and explore
        (
            "DISCOVERY",
            "I'm looking for a laptop for software development and "
            "some gaming. What's available?",
        ),
        # Turn 2 — still DISCOVERY: capture preferences (triggers → CURATION)
        (
            "DISCOVERY → CURATION",
            "Good options. Please note my preferences: under $2000, "
            "16GB+ RAM, and good battery life.",
        ),
        # Turn 3 — CURATION: filter and compare
        (
            "CURATION",
            "Now filter those to match my preferences and compare "
            "the top options.",
        ),
        # Turn 4 — still CURATION: shortlist (triggers → RECOMMENDATION)
        (
            "CURATION → RECOMMENDATION",
            "Narrow it down to a shortlist of the top 2-3.",
        ),
        # Turn 5 — RECOMMENDATION: present final pick (triggers → COMPLETE)
        (
            "RECOMMENDATION → COMPLETE",
            "Great. Give me your final recommendation.",
        ),
    ]

    for i, (label, user_msg) in enumerate(turns, 1):
        conversation.append(HumanMessage(content=user_msg))
        result = agent.invoke(  # type: ignore[call-overload]
            {"messages": conversation},
            config={"run_name": f"BSM: Phase Steering ({label})"},
        )
        _print_turn(i, label, user_msg, result, middleware)
        conversation = list(result["messages"])

    print("=" * 64)
    print("DEMO COMPLETE")
    print(f"Final phase: {middleware.current_phase.name}")
    transitions = [(p.name, itr) for p, itr in middleware.phase_history]
    print(f"Phase transitions: {transitions}")
    print("=" * 64)


def main() -> None:
    """Entry point for the demo."""
    run_demo()


if __name__ == "__main__":
    main()
