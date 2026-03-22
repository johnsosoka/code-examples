"""Demonstration of Phase-Based Steering middleware with a real LLM.

Invokes the agent once per phase (GATHER -> ANALYZE -> RECOMMEND -> COMPLETE)
to show how the middleware guides the agent through a structured research workflow.

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

from bsm_examples.phase_steering.config import (  # noqa: E402
    PhaseBehavior,
    PhaseConfig,
    PhaseTransition,
)
from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware  # noqa: E402
from bsm_examples.phase_steering.phase import Phase  # noqa: E402
from bsm_examples.phase_steering.state import PhaseAgentState  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@tool
def search(query: str) -> str:
    """Search for information on a topic. Returns relevant findings."""
    return f"Found results for '{query}': [relevant documentation and best practices]"


@tool
def fetch_document(url: str) -> str:
    """Fetch and summarize a document from a URL."""
    return f"Document from {url}: [summary of key points about the topic]"


@tool
def compare(items: str) -> str:
    """Compare multiple approaches or items. Pass comma-separated items."""
    item_list = [i.strip() for i in items.split(",")]
    return (
        f"Comparison of {len(item_list)} items: "
        f"[trade-off analysis for {', '.join(item_list)}]"
    )


@tool
def synthesize(analysis: str) -> str:
    """Synthesize analysis into actionable recommendations."""
    return "Based on analysis: [3 prioritized recommendations with justification]"


def create_research_config() -> PhaseConfig:
    """Create the PhaseConfig for a research assistant workflow.

    Defines GATHER, ANALYZE, RECOMMEND, and COMPLETE phase behaviors
    and the valid transitions between them.
    """
    return PhaseConfig(
        phases={
            Phase.GATHER: PhaseBehavior(
                system_prompt_addition=(
                    "Focus on collecting comprehensive information about the topic. "
                    "Search for current standards, best practices, and relevant sources. "
                    "Do not analyze or make recommendations yet — just gather raw information."
                ),
                allowed_tools=["search", "fetch_document"],
                completion_criteria="Collect at least 3 relevant sources",
                max_iterations=5,
            ),
            Phase.ANALYZE: PhaseBehavior(
                system_prompt_addition=(
                    "Analyze the gathered information for patterns, trade-offs, and key insights. "
                    "Compare different approaches and evaluate their suitability. "
                    "Do not make final recommendations yet — focus on understanding the landscape."
                ),
                allowed_tools=["compare"],
                completion_criteria="Produce substantive analysis notes (100+ chars)",
                max_iterations=3,
            ),
            Phase.RECOMMEND: PhaseBehavior(
                system_prompt_addition=(
                    "Based on your analysis, synthesize clear, actionable recommendations. "
                    "Provide specific guidance for different scenarios and justify your "
                    "recommendations with evidence from the analysis."
                ),
                allowed_tools=["synthesize"],
                completion_criteria="Generate at least one concrete recommendation",
                max_iterations=2,
            ),
            Phase.COMPLETE: PhaseBehavior(
                system_prompt_addition=(
                    "Present the final recommendations to the user in a clear, structured format. "
                    "Summarize the research process and key findings."
                ),
                allowed_tools=[],
                completion_criteria="Workflow complete",
                max_iterations=1,
            ),
        },
        transitions=[
            PhaseTransition(
                from_phase=Phase.GATHER,
                to_phase=Phase.ANALYZE,
                condition_description="At least 3 data items collected and 3+ iterations",
            ),
            PhaseTransition(
                from_phase=Phase.ANALYZE,
                to_phase=Phase.RECOMMEND,
                condition_description="Analysis notes are substantive (100+ chars) and 2+ iters",
            ),
            PhaseTransition(
                from_phase=Phase.RECOMMEND,
                to_phase=Phase.COMPLETE,
                condition_description="At least one recommendation generated",
            ),
        ],
        initial_phase=Phase.GATHER,
        enable_auto_transition=True,
    )


def run_demo() -> None:
    """Run a research assistant session that progresses through all four phases.

    Because the middleware tracks phase state internally, the agent is invoked
    once per phase with a targeted prompt. After enough iterations, the
    middleware transitions the agent into the next phase automatically.
    """
    query = "What are the best practices for API authentication in 2024?"

    print("=" * 64)
    print("PHASE-BASED STEERING DEMO: Research Assistant")
    print("=" * 64)
    print(f"Query: {query}")
    print()

    config = create_research_config()
    middleware = PhaseSteeringMiddleware(config)

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4.1-mini"),
        tools=[search, fetch_document, compare, synthesize],
        system_prompt=(
            "You are a research assistant. "
            "You will receive <system> tags with phase guidance "
            "— follow them and use only the tools specified."
        ),
        middleware=[middleware],
        state_schema=PhaseAgentState,
    )

    # Each invocation drives one phase. The middleware accumulates iteration counts
    # and auto-transitions when completion criteria are met.
    phase_prompts = [
        # GATHER phase — ask for broad information collection
        (
            f"{query} Search for current best practices, relevant RFCs, "
            "and security standards. Use the search and fetch_document tools."
        ),
        # ANALYZE phase — ask for comparison and trade-off analysis
        (
            "Compare the main API authentication approaches you found: "
            "OAuth 2.0, API keys, and JWT. Use the compare tool."
        ),
        # RECOMMEND phase — ask for synthesis into recommendations
        (
            "Based on the comparison, synthesize your top recommendations "
            "for API authentication. Use the synthesize tool."
        ),
        # COMPLETE phase — ask for a structured final summary
        (
            "Present a final structured summary of your research findings "
            "and recommendations."
        ),
    ]

    messages: list[HumanMessage] = []

    for i, prompt in enumerate(phase_prompts, 1):
        print(f"--- Invocation {i} ---")
        print(f"User: {prompt[:80]}...")

        messages.append(HumanMessage(content=prompt))
        result = agent.invoke({"messages": messages})  # type: ignore[call-overload]

        current_phase = result.get("current_phase", "UNKNOWN")
        phase_iteration = result.get("phase_iteration", 0)
        total_iterations = result.get("total_iterations", 0)

        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content[:200]}")
        print(
            f"  current_phase={current_phase}  "
            f"phase_iteration={phase_iteration}  total_iterations={total_iterations}"
        )
        print()

        messages = list(result["messages"])

    print("=" * 64)
    print("DEMO COMPLETE")
    print(f"Phase history: {[p.name for p, _ in middleware.phase_history]}")
    print("=" * 64)
    print()


def main() -> None:
    """Entry point for the demo."""
    run_demo()


if __name__ == "__main__":
    main()
