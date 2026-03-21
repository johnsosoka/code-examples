"""Phase detection strategies for identifying current workflow phase.

This module provides abstract base class and concrete implementations
for detecting the current phase based on state, messages, and tool usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from .config import PhaseConfig
    from .phase import Phase
    from .state import PhaseState


class PhaseDetector(ABC):
    """Abstract base class for phase detection strategies.

    Implementations analyze agent state, message history, and tool usage
    to determine the current operational phase of the workflow.

    Example:
        >>> class CustomDetector(PhaseDetector):
        ...     def detect_phase(self, state, messages, last_tool_calls=None):
        ...         return Phase.GATHER
        ...     def should_transition(self, current_phase, state, messages):
        ...         return False, None
    """

    def __init__(self, config: PhaseConfig) -> None:
        """Initialize the detector with phase configuration.

        Args:
            config: Phase configuration defining behaviors and transitions.
        """
        self.config = config

    @abstractmethod
    def detect_phase(
        self,
        state: PhaseState,
        messages: list[BaseMessage],
        last_tool_calls: list[dict[str, Any]] | None = None,
    ) -> Phase:
        """Analyze current state and determine the appropriate phase.

        Args:
            state: Current workflow state with phase tracking.
            messages: Message history for context analysis.
            last_tool_calls: Recent tool invocations if any.

        Returns:
            Detected phase based on heuristics and state analysis.
        """
        raise NotImplementedError

    @abstractmethod
    def should_transition(
        self,
        current_phase: Phase,
        state: PhaseState,
        messages: list[BaseMessage],
    ) -> tuple[bool, Phase | None]:
        """Determine if a phase transition should occur.

        Args:
            current_phase: Currently active phase.
            state: Current workflow state.
            messages: Recent message history.

        Returns:
            Tuple of (should_transition, target_phase_or_none).
        """
        raise NotImplementedError


class DefaultPhaseDetector(PhaseDetector):
    """Default implementation using heuristics based on tool usage and state content.

    Detection strategy:
    1. Check for explicit phase markers in state
    2. Analyze tool usage patterns
    3. Examine message content for phase indicators
    4. Apply iteration limits from phase config

    Example:
        >>> config = PhaseConfig(phases={...})
        >>> detector = DefaultPhaseDetector(config)
        >>> phase = detector.detect_phase(state, messages)
    """

    # Tool name patterns that indicate specific phases
    GATHER_TOOLS: set[str] = {"search", "fetch", "retrieve", "query", "lookup", "find"}
    ANALYZE_TOOLS: set[str] = {"compare", "evaluate", "score", "rank", "analyze", "assess"}
    RECOMMEND_TOOLS: set[str] = {"synthesize", "recommend", "generate_report", "summarize"}

    def detect_phase(
        self,
        state: PhaseState,
        messages: list[BaseMessage],
        last_tool_calls: list[dict[str, Any]] | None = None,
    ) -> Phase:
        """Detect phase using heuristic analysis.

        Priority order:
        1. Explicit phase in state.current_phase
        2. Recent tool call patterns
        3. State content indicators
        4. Default to initial phase

        Args:
            state: Current workflow state.
            messages: Message history.
            last_tool_calls: Recent tool invocations.

        Returns:
            Detected phase.
        """
        from .phase import Phase

        # Priority 1: Check for explicit phase in state
        if state.current_phase is not None:
            return state.current_phase

        # Priority 2: Analyze recent tool calls
        if last_tool_calls:
            tool_names: set[str] = {tc.get("name", "").lower() for tc in last_tool_calls}

            if tool_names & self.RECOMMEND_TOOLS:
                return Phase.RECOMMEND
            elif tool_names & self.ANALYZE_TOOLS:
                return Phase.ANALYZE
            elif tool_names & self.GATHER_TOOLS:
                return Phase.GATHER

        # Priority 3: Check state content for phase indicators
        if state.recommendations:
            return Phase.RECOMMEND
        elif state.analysis_notes:
            return Phase.ANALYZE
        elif state.gathered_data:
            return Phase.GATHER

        # Default to initial phase
        return self.config.initial_phase if self.config.initial_phase is not None else Phase.GATHER

    def should_transition(
        self,
        current_phase: Phase,
        state: PhaseState,
        messages: list[BaseMessage],
    ) -> tuple[bool, Phase | None]:
        """Determine if phase should transition based on completion criteria.

        Checks iteration limits, transition signals in messages, and
        phase-specific completion criteria.

        Args:
            current_phase: Currently active phase.
            state: Current workflow state.
            messages: Recent message history.

        Returns:
            Tuple of (should_transition, target_phase_or_none).
        """
        phase_config = self.config.phases.get(current_phase)
        if not phase_config:
            return False, None

        # Check iteration limit
        phase_iterations = state.phase_iterations.get(current_phase, 0)
        if phase_iterations >= phase_config.max_iterations:
            # Force transition to next logical phase
            next_phase = self._get_next_phase(current_phase)
            return True, next_phase

        # Check for explicit transition signals in messages
        if messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", "")
            if isinstance(content, str):
                transition_signal = self._detect_transition_signal(content, current_phase)
                if transition_signal:
                    return True, transition_signal

        # Check state-based completion criteria
        if self._phase_completion_criteria_met(current_phase, state):
            next_phase = self._get_next_phase(current_phase)
            return True, next_phase

        return False, None

    def _detect_transition_signal(self, content: str, current_phase: Phase) -> Phase | None:
        """Detect explicit transition requests in message content.

        Args:
            content: Message content to analyze.
            current_phase: Current phase to check signals for.

        Returns:
            Target phase if signal detected, None otherwise.
        """
        from .phase import Phase

        content_lower = content.lower()

        # Signals indicating readiness to move forward
        transition_signals: dict[Phase, list[str]] = {
            Phase.GATHER: ["ready to analyze", "let's analyze", "moving to analysis"],
            Phase.ANALYZE: ["ready to recommend", "my recommendation", "final recommendation"],
        }

        signals = transition_signals.get(current_phase, [])
        if any(signal in content_lower for signal in signals):
            return self._get_next_phase(current_phase)

        return None

    def _phase_completion_criteria_met(self, phase: Phase, state: PhaseState) -> bool:
        """Check if current phase's completion criteria are satisfied.

        Args:
            phase: Phase to check completion for.
            state: Current workflow state.

        Returns:
            True if completion criteria are met.
        """
        from .phase import Phase

        if phase == Phase.GATHER:
            # GATHER complete when we have data and iteration count suggests saturation
            return len(state.gathered_data) >= 3 and state.phase_iterations.get(phase, 0) >= 3
        elif phase == Phase.ANALYZE:
            # ANALYZE complete when analysis notes exist and are substantive
            return len(state.analysis_notes) >= 100 and state.phase_iterations.get(phase, 0) >= 2
        elif phase == Phase.RECOMMEND:
            # RECOMMEND complete when recommendations exist
            return len(state.recommendations) > 0

        return False

    def _get_next_phase(self, current: Phase) -> Phase:
        """Get the logical next phase in sequence.

        Args:
            current: Current phase.

        Returns:
            Next phase in the standard sequence.
        """
        from .phase import Phase

        phase_order = [Phase.GATHER, Phase.ANALYZE, Phase.RECOMMEND, Phase.COMPLETE]
        try:
            idx = phase_order.index(current)
            return phase_order[min(idx + 1, len(phase_order) - 1)]
        except ValueError:
            return Phase.COMPLETE
