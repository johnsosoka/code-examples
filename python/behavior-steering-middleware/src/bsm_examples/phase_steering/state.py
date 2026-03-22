"""State type definition for phase-based workflow tracking.

This module defines the PhaseState model for tracking workflow progress
through different phases, compatible with LangGraph's StateGraph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .phase import Phase


class PhaseState(BaseModel):
    """Internal state for tracking phase-based workflow progress.

    Used by PhaseSteeringMiddleware for rich phase bookkeeping. Not a
    LangGraph state schema — use PhaseAgentState for agent state integration.

    Attributes:
        current_phase: Currently active phase (None = auto-detect).
        phase_iterations: Iteration count per phase.
        phase_history: Ordered history of phase transitions.
        gathered_data: Data collected during GATHER phase.
        analysis_notes: Analysis output from ANALYZE phase.
        recommendations: Final recommendations from RECOMMEND phase.
        query: Original user query or task description.
        metadata: Additional workflow metadata.

    Example:
        >>> state = PhaseState(query="Research best practices")
        >>> state.record_phase_entry(Phase.GATHER)
        >>> state.add_gathered_data({"source": "docs", "content": "..."})
    """

    # Phase tracking fields
    current_phase: Phase | None = Field(
        default=None,
        description="Currently active phase (None = auto-detect)",
    )
    phase_iterations: dict[Phase, int] = Field(
        default_factory=dict,
        description="Iteration count per phase",
    )
    phase_history: list[Phase] = Field(
        default_factory=list,
        description="Ordered history of phase transitions",
    )

    # Domain-specific data fields (customize per use case)
    gathered_data: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Data collected during GATHER phase",
    )
    analysis_notes: str = Field(
        default="",
        description="Analysis output from ANALYZE phase",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Final recommendations from RECOMMEND phase",
    )

    # Metadata
    query: str = Field(
        default="",
        description="Original user query or task description",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional workflow metadata",
    )

    def record_phase_entry(self, phase: Phase) -> None:
        """Record entry into a new phase.

        Updates the phase history and sets the current phase.
        Also increments the iteration count for the phase.

        Args:
            phase: The phase being entered.

        Example:
            >>> state = PhaseState()
            >>> state.record_phase_entry(Phase.GATHER)
            >>> state.current_phase
            <Phase.GATHER: 1>
        """
        if self.current_phase is not None:
            self.phase_history.append(self.current_phase)
        self.current_phase = phase
        self.phase_iterations[phase] = self.phase_iterations.get(phase, 0) + 1

    def increment_iteration(self, phase: Phase | None = None) -> None:
        """Increment iteration count for specified or current phase.

        Args:
            phase: The phase to increment (defaults to current_phase).

        Example:
            >>> state = PhaseState(current_phase=Phase.GATHER)
            >>> state.increment_iteration()
            >>> state.phase_iterations[Phase.GATHER]
            1
        """
        target = phase or self.current_phase
        if target:
            self.phase_iterations[target] = self.phase_iterations.get(target, 0) + 1

    def add_gathered_data(self, data: dict[str, Any]) -> None:
        """Add data collected during gathering phase.

        Args:
            data: Dictionary containing collected data.

        Example:
            >>> state = PhaseState()
            >>> state.add_gathered_data({"source": "web", "title": "..."})
        """
        self.gathered_data.append(data)

    def set_analysis(self, notes: str) -> None:
        """Set analysis notes from analysis phase.

        Args:
            notes: Analysis output string.

        Example:
            >>> state = PhaseState()
            >>> state.set_analysis("Key findings: ...")
        """
        self.analysis_notes = notes

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation from recommendation phase.

        Args:
            recommendation: Recommendation text to add.

        Example:
            >>> state = PhaseState()
            >>> state.add_recommendation("Use OAuth 2.0 for security.")
        """
        self.recommendations.append(recommendation)

    def is_phase_complete(self, phase: Phase, min_iterations: int = 1) -> bool:
        """Check if a phase has been visited with minimum iterations.

        Args:
            phase: The phase to check.
            min_iterations: Minimum required iterations (default 1).

        Returns:
            True if the phase has been visited at least min_iterations times.

        Example:
            >>> state = PhaseState(phase_iterations={Phase.GATHER: 3})
            >>> state.is_phase_complete(Phase.GATHER, min_iterations=2)
            True
        """
        return self.phase_iterations.get(phase, 0) >= min_iterations

    @property
    def total_iterations(self) -> int:
        """Total iterations across all phases.

        Returns:
            Sum of all phase iteration counts.

        Example:
            >>> state = PhaseState(phase_iterations={Phase.GATHER: 3, Phase.ANALYZE: 2})
            >>> state.total_iterations
            5
        """
        return sum(self.phase_iterations.values())


# Rebuild the model after Phase is available — deferred to break circular import
from .phase import Phase  # noqa: E402

PhaseState.model_rebuild()


# PhaseAgentState imports deferred so PhaseState.model_rebuild() runs first
from langchain.agents.middleware import AgentState  # noqa: E402
from typing_extensions import NotRequired  # noqa: E402


class PhaseAgentState(AgentState):
    """Agent state schema with phase tracking fields pushed by middleware.

    This TypedDict extends AgentState for use with LangGraph's create_agent.
    The middleware writes these fields via ExtendedModelResponse Command updates
    so callers can observe phase progression in the agent state.

    Note:
        PhaseState (above) is a richer Pydantic model used for internal
        middleware bookkeeping. PhaseAgentState is the lightweight TypedDict
        that surfaces key phase info into the public agent state.
    """

    current_phase: NotRequired[str]
    phase_iteration: NotRequired[int]
    total_iterations: NotRequired[int]
