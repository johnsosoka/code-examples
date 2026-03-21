"""Demonstration of Phase-Based Steering middleware.

This module provides a runnable demonstration of the phase-based steering
pattern using a research assistant workflow example.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from .config import PhaseConfig
    from .middleware import PhaseSteeringMiddleware
    from .state import PhaseState

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_research_config() -> PhaseConfig:
    """Create phase configuration for the research assistant demo.

    Returns:
        PhaseConfig with GATHER, ANALYZE, RECOMMEND, and COMPLETE phases.
    """
    from .config import PhaseBehavior, PhaseConfig, PhaseTransition
    from .phase import Phase

    return PhaseConfig(
        phases={
            Phase.GATHER: PhaseBehavior(
                system_prompt_addition=(
                    "Focus on collecting comprehensive information about the topic. "
                    "Search for current standards, best practices, and relevant sources. "
                    "Do not analyze or make recommendations yet—just gather raw information."
                ),
                allowed_tools=["search", "fetch_document", "query"],
                completion_criteria="Collect at least 3 relevant sources",
                max_iterations=5,
            ),
            Phase.ANALYZE: PhaseBehavior(
                system_prompt_addition=(
                    "Analyze the gathered information for patterns, trade-offs, and key insights. "
                    "Compare different approaches and evaluate their suitability. "
                    "Do not make final recommendations yet—focus on understanding the landscape."
                ),
                allowed_tools=["compare", "evaluate", "score"],
                completion_criteria="Produce substantive analysis notes (100+ chars)",
                max_iterations=3,
            ),
            Phase.RECOMMEND: PhaseBehavior(
                system_prompt_addition=(
                    "Based on your analysis, synthesize clear, actionable recommendations. "
                    "Provide specific guidance for different scenarios and justify your "
                    "recommendations with evidence from the analysis."
                ),
                allowed_tools=["synthesize", "generate_report"],
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
                condition_description="Analysis notes are substantive (100+ chars) and 2+ iterations",
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


def mock_search(query: str) -> dict[str, Any]:
    """Mock search tool for demonstration.

    Args:
        query: Search query string.

    Returns:
        Mock search result.
    """
    return {
        "tool": "search",
        "query": query,
        "results": [
            {"title": f"Result for: {query}", "source": "docs.example.com", "relevance": 0.95}
        ],
    }


def mock_compare(approaches: list[str], criteria: list[str]) -> dict[str, Any]:
    """Mock compare tool for demonstration.

    Args:
        approaches: List of approaches to compare.
        criteria: List of criteria to compare against.

    Returns:
        Mock comparison result.
    """
    return {
        "tool": "compare",
        "approaches": approaches,
        "criteria": criteria,
        "comparison": f"Compared {len(approaches)} approaches across {len(criteria)} criteria",
    }


def mock_synthesize(analysis: str, format_type: str) -> dict[str, Any]:
    """Mock synthesize tool for demonstration.

    Args:
        analysis: Analysis text to synthesize.
        format_type: Output format type.

    Returns:
        Mock synthesis result.
    """
    return {
        "tool": "synthesize",
        "format": format_type,
        "recommendations": [
            "Recommendation 1: Use OAuth 2.0 for high-security APIs",
            "Recommendation 2: Use JWT for internal microservices",
            "Recommendation 3: Use API keys for simple public APIs",
        ],
    }


def run_gather_phase(
    middleware: PhaseSteeringMiddleware,
    state: PhaseState,
    query: str,
) -> None:
    """Run the GATHER phase of the workflow.

    Args:
        middleware: Phase steering middleware instance.
        state: Current workflow state.
        query: User's research query.
    """
    from .phase import Phase

    print(f"\n[Phase: {Phase.GATHER.display_name}] Starting information gathering...")

    # Simulate gathering iterations
    search_queries = [
        f"{query} best practices 2024",
        f"{query} standards and security considerations",
        f"{query} comparison of approaches",
    ]

    for i, search_query in enumerate(search_queries, 1):
        print(f"\n  Iteration {i}/5")
        print(f"  → Searching for: {search_query}")

        # Mock tool call
        result = mock_search(search_query)
        state.add_gathered_data(result)

        # Process through middleware
        messages: list[BaseMessage] = [HumanMessage(content=f"Search results for: {search_query}")]
        phase, modified_messages = middleware.process_request(state, messages)

        # Show injected guidance
        system_msg = next((m for m in modified_messages if isinstance(m, SystemMessage)), None)
        if system_msg and isinstance(system_msg.content, str):
            guidance_lines = system_msg.content.split("\n")
            print(f"  → Phase guidance: {guidance_lines[0]}")

        print(f"  → Found: {len(result['results'])} relevant source(s)")

        # Check if we should transition
        if phase != Phase.GATHER:
            print(f"\n--- Phase Transition: {Phase.GATHER.display_name} → {phase.display_name} ---")
            break

    print(f"\n  Total gathered: {len(state.gathered_data)} data items")


def run_analyze_phase(
    middleware: PhaseSteeringMiddleware,
    state: PhaseState,
) -> None:
    """Run the ANALYZE phase of the workflow.

    Args:
        middleware: Phase steering middleware instance.
        state: Current workflow state.
    """
    from .phase import Phase

    print(f"\n[Phase: {Phase.ANALYZE.display_name}] Starting analysis...")

    # Simulate analysis iterations
    analyses = [
        "Initial analysis: Multiple authentication approaches identified. "
        "OAuth 2.0 provides highest security but adds complexity. "
        "JWT balances security and simplicity for internal APIs.",
        "Deep dive: API keys simplest but least secure. "
        "OAuth 2.0 with PKCE recommended for public clients. "
        "JWT with short expiration suitable for microservices.",
    ]

    for i, analysis in enumerate(analyses, 1):
        print(f"\n  Iteration {i}/3")

        if i == 1:
            # Compare approaches
            print("  → Comparing authentication approaches...")
            result = mock_compare(
                approaches=["OAuth 2.0", "JWT", "API Keys"],
                criteria=["security", "complexity", "use_case_fit"],
            )
            print(f"  → {result['comparison']}")

        # Set analysis notes
        state.set_analysis(analysis)
        print(f"  → Analysis: {analysis[:80]}...")

        # Process through middleware
        messages: list[BaseMessage] = [HumanMessage(content=f"Analysis iteration {i}: {analysis}")]
        phase, modified_messages = middleware.process_request(state, messages)

        # Show injected guidance
        system_msg = next((m for m in modified_messages if isinstance(m, SystemMessage)), None)
        if system_msg and isinstance(system_msg.content, str):
            guidance_lines = system_msg.content.split("\n")
            print(f"  → Phase guidance: {guidance_lines[0]}")

        # Check if we should transition
        if phase != Phase.ANALYZE:
            print(f"\n--- Phase Transition: {Phase.ANALYZE.display_name} → {phase.display_name} ---")
            break

    print(f"\n  Analysis notes length: {len(state.analysis_notes)} characters")


def run_recommend_phase(
    middleware: PhaseSteeringMiddleware,
    state: PhaseState,
) -> None:
    """Run the RECOMMEND phase of the workflow.

    Args:
        middleware: Phase steering middleware instance.
        state: Current workflow state.
    """
    from .phase import Phase

    print(f"\n[Phase: {Phase.RECOMMEND.display_name}] Synthesizing recommendations...")

    print("\n  Iteration 1/2")
    print("  → Synthesizing recommendations from analysis...")

    # Mock synthesis
    result = mock_synthesize(analysis=state.analysis_notes, format_type="recommendations")

    for rec in result["recommendations"]:
        state.add_recommendation(rec)

    # Process through middleware
    messages: list[BaseMessage] = [HumanMessage(content="Please synthesize final recommendations.")]
    phase, modified_messages = middleware.process_request(state, messages)

    # Show injected guidance
    system_msg = next((m for m in modified_messages if isinstance(m, SystemMessage)), None)
    if system_msg and isinstance(system_msg.content, str):
        guidance_lines = system_msg.content.split("\n")
        print(f"  → Phase guidance: {guidance_lines[0]}")

    print(f"  → Generated {len(result['recommendations'])} recommendations")

    # Check if we should transition
    if phase != Phase.RECOMMEND:
        print(f"\n--- Phase Transition: {Phase.RECOMMEND.display_name} → {phase.display_name} ---")


def run_complete_phase(state: PhaseState) -> None:
    """Run the COMPLETE phase and display final results.

    Args:
        state: Current workflow state.
    """
    from .phase import Phase

    print(f"\n{'=' * 64}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'=' * 64}")

    for i, rec in enumerate(state.recommendations, 1):
        print(f"\n{i}. {rec}")

    print(f"\n{'=' * 64}")
    print("PHASE HISTORY")
    print(f"{'=' * 64}")

    # Build phase history string
    history_parts: list[str] = []
    for phase in Phase:
        count = state.phase_iterations.get(phase, 0)
        if count > 0:
            history_parts.append(f"{phase.display_name}({count})")

    print(f"Phase progression: {' → '.join(history_parts)}")
    print(f"Total iterations: {state.total_iterations}")
    print(f"{'=' * 64}")


def run_demo(query: str | None = None) -> None:
    """Run the complete phase-based steering demonstration.

    This function demonstrates a multi-phase research assistant workflow
    that progresses through GATHER → ANALYZE → RECOMMEND → COMPLETE phases.

    Args:
        query: Optional custom research query (defaults to API authentication topic).

    Example:
        >>> run_demo()
        >>> run_demo("What are the best practices for database sharding?")
    """
    from .middleware import PhaseSteeringMiddleware
    from .phase import Phase
    from .state import PhaseState

    # Default query if none provided
    if query is None:
        query = "What are the best practices for API authentication in 2024?"

    print(f"{'=' * 64}")
    print("Phase-Based Steering Demo: Research Assistant")
    print(f"{'=' * 64}")
    print(f"\nUser Query: {query}")

    # Initialize configuration and middleware
    config = create_research_config()
    middleware = PhaseSteeringMiddleware(config)

    # Initialize state
    state = PhaseState(query=query)
    state.record_phase_entry(Phase.GATHER)

    # Run through phases
    run_gather_phase(middleware, state, query)
    run_analyze_phase(middleware, state)
    run_recommend_phase(middleware, state)

    # Record completion
    state.record_phase_entry(Phase.COMPLETE)
    run_complete_phase(state)


def main() -> None:
    """Main entry point for the demo."""
    import sys

    # Allow custom query from command line
    query = None
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    run_demo(query)


if __name__ == "__main__":
    main()
