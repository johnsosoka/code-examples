"""Phase steering middleware for LangGraph agents.

This module provides the PhaseSteeringMiddleware class that injects
phase-appropriate behavior guidance into agent execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from .config import PhaseBehavior, PhaseConfig
    from .detector import PhaseDetector
    from .phase import Phase
    from .state import PhaseState

logger = logging.getLogger(__name__)


class PhaseSteeringMiddleware:
    """Middleware that injects phase-appropriate behavior guidance into agent execution.

    This middleware implements the Phase-Based Steering pattern by:
    1. Detecting the current operational phase from state and context
    2. Injecting phase-specific system prompt additions
    3. Validating and executing phase transitions
    4. Tracking phase iteration counts for progression management

    The middleware wraps model calls to add behavioral context without modifying
    the underlying agent logic, maintaining clean separation of concerns.

    Example:
        >>> from .config import PhaseConfig, PhaseBehavior
        >>> from .phase import Phase
        >>> config = PhaseConfig(
        ...     phases={
        ...         Phase.GATHER: PhaseBehavior(
        ...             system_prompt_addition="Focus on collecting comprehensive information...",
        ...             allowed_tools=["search", "fetch_document"],
        ...             max_iterations=5,
        ...         ),
        ...         Phase.ANALYZE: PhaseBehavior(
        ...             system_prompt_addition="Analyze the gathered data for patterns...",
        ...             allowed_tools=["compare", "evaluate"],
        ...             max_iterations=3,
        ...         ),
        ...     },
        ...     transitions=[...],
        ... )
        >>> middleware = PhaseSteeringMiddleware(config)
    """

    def __init__(
        self,
        config: PhaseConfig,
        detector: PhaseDetector | None = None,
    ) -> None:
        """Initialize the phase steering middleware.

        Args:
            config: Phase configuration defining behaviors and transitions.
            detector: Optional custom phase detector (uses DefaultPhaseDetector if None).
        """
        self.config = config
        # Import here to avoid circular dependency
        from .detector import DefaultPhaseDetector

        self.detector = detector or DefaultPhaseDetector(config)
        self._current_phase: Phase = config.initial_phase if config.initial_phase is not None else Phase.GATHER
        self._phase_history: list[tuple[Phase, int]] = []  # (phase, iteration)
        self._phase_iterations: dict[Phase, int] = {}

    def process_request(
        self,
        state: PhaseState,
        messages: list[BaseMessage],
    ) -> tuple[Phase, list[BaseMessage]]:
        """Process a request to inject phase-appropriate guidance.

        This method:
        1. Detects current phase from state/messages
        2. Checks for and executes phase transitions
        3. Injects phase-specific system prompt additions
        4. Tracks phase iteration counts

        Args:
            state: Current workflow state.
            messages: Current message list.

        Returns:
            Tuple of (detected_phase, modified_messages).
        """
        # Detect current phase
        detected_phase = self.detector.detect_phase(state, messages)

        # Check for phase transition
        should_transition, target_phase = self.detector.should_transition(
            detected_phase, state, messages
        )

        if should_transition and target_phase and self.config.enable_auto_transition:
            logger.info(f"Phase transition: {detected_phase.name} -> {target_phase.name}")
            self._record_phase_transition(detected_phase, target_phase)
            detected_phase = target_phase
            state.record_phase_entry(detected_phase)

        self._current_phase = detected_phase

        # Update phase iteration count
        self._increment_phase_iteration(detected_phase)
        state.increment_iteration(detected_phase)

        # Inject phase-specific guidance
        modified_messages = self._inject_phase_guidance(messages, detected_phase)

        # Log phase context
        phase_config = self.config.phases.get(detected_phase)
        if phase_config:
            logger.debug(
                f"Executing in {detected_phase.name} phase "
                f"(iteration {self._get_phase_iteration(detected_phase)})"
            )

        return detected_phase, modified_messages

    def _inject_phase_guidance(
        self,
        messages: list[BaseMessage],
        phase: Phase,
    ) -> list[BaseMessage]:
        """Inject phase-specific system prompt additions into messages.

        Args:
            messages: Original message list.
            phase: Current phase to inject guidance for.

        Returns:
            Modified message list with phase guidance.
        """
        phase_config = self.config.phases.get(phase)
        if not phase_config or not phase_config.system_prompt_addition:
            return list(messages)

        # Find existing system message or determine insertion point
        messages_out = list(messages)
        system_msg_idx: int | None = None

        for i, msg in enumerate(messages_out):
            if isinstance(msg, SystemMessage):
                system_msg_idx = i
                break

        # Build phase guidance message
        phase_guidance = self._build_phase_guidance(phase, phase_config)

        if system_msg_idx is not None:
            # Append to existing system message
            existing_msg = messages_out[system_msg_idx]
            existing_content = existing_msg.content
            if isinstance(existing_content, str):
                new_content = f"{existing_content}\n\n{phase_guidance}"
                messages_out[system_msg_idx] = SystemMessage(content=new_content)
        else:
            # Insert new system message at the beginning
            messages_out.insert(0, SystemMessage(content=phase_guidance))

        return messages_out

    def _build_phase_guidance(self, phase: Phase, config: PhaseBehavior) -> str:
        """Build the phase guidance string to inject.

        Args:
            phase: Current phase.
            config: Phase behavior configuration.

        Returns:
            Formatted guidance string.
        """
        guidance_parts: list[str] = [
            f"[PHASE: {phase.display_name}]",
            config.system_prompt_addition,
        ]

        if config.allowed_tools:
            guidance_parts.append(
                f"Available tools in this phase: {', '.join(config.allowed_tools)}"
            )

        if config.completion_criteria:
            guidance_parts.append(f"Phase completion criteria: {config.completion_criteria}")

        return "\n".join(guidance_parts)

    def _record_phase_transition(self, from_phase: Phase, to_phase: Phase) -> None:
        """Record a phase transition in history.

        Args:
            from_phase: Phase being transitioned from.
            to_phase: Phase being transitioned to.
        """
        self._phase_history.append((from_phase, self._get_phase_iteration(from_phase)))
        logger.info(f"Transitioned from {from_phase.name} to {to_phase.name}")

    def _increment_phase_iteration(self, phase: Phase) -> None:
        """Increment iteration count for the given phase.

        Args:
            phase: Phase to increment count for.
        """
        self._phase_iterations[phase] = self._phase_iterations.get(phase, 0) + 1

    def _get_phase_iteration(self, phase: Phase) -> int:
        """Get current iteration count for the given phase.

        Args:
            phase: Phase to get count for.

        Returns:
            Current iteration count (0 if not yet visited).
        """
        return self._phase_iterations.get(phase, 0)

    @property
    def current_phase(self) -> Phase:
        """Get the currently active phase."""
        return self._current_phase

    @property
    def phase_history(self) -> list[tuple[Phase, int]]:
        """Get history of phase transitions with iteration counts.

        Returns:
            Copy of the phase history list.
        """
        return self._phase_history.copy()

    def reset(self) -> None:
        """Reset middleware state for a new conversation."""
        from .phase import Phase

        self._current_phase = self.config.initial_phase if self.config.initial_phase is not None else Phase.GATHER
        self._phase_history = []
        self._phase_iterations = {}
        logger.debug("Phase steering middleware reset")

    def should_continue(self, state: PhaseState) -> bool:
        """Check if the workflow should continue or is complete.

        Args:
            state: Current workflow state.

        Returns:
            False if in COMPLETE phase, True otherwise.
        """
        from .phase import Phase

        return state.current_phase != Phase.COMPLETE
