"""Phase steering middleware for LangGraph agents.

This module provides the PhaseSteeringMiddleware class that injects
phase-appropriate behavior guidance into agent execution using the official
LangChain AgentMiddleware API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from .state import PhaseAgentState, PhaseState

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest

    from .config import PhaseBehavior, PhaseConfig
    from .detector import PhaseDetector
    from .phase import Phase

logger = logging.getLogger(__name__)


class PhaseSteeringMiddleware(AgentMiddleware):
    """Middleware that guides agent execution through workflow phases.

    Implements the Phase-Based Steering pattern using the AgentMiddleware API:
    - ``wrap_model_call``: detects phase, appends ``<system>``-tagged guidance
      to the message stream, and reports phase state via ``ExtendedModelResponse``.
    - ``wrap_tool_call``: enforces per-phase tool allow-lists, blocking tools
      not permitted in the current phase.

    Steering content is injected into the conversation messages — not the system
    prompt. The agent's system prompt should mention that ``<system>`` tags will
    appear with phase guidance.

    Internal tracking is done via a ``PhaseState`` Pydantic object. A lightweight
    ``PhaseAgentState`` TypedDict schema surfaces phase info into the agent state.

    Example:
        >>> from langchain.agents import create_agent
        >>> from langchain_openai import ChatOpenAI
        >>> config = PhaseConfig(phases={Phase.GATHER: PhaseBehavior(...)})
        >>> middleware = PhaseSteeringMiddleware(config)
        >>> agent = create_agent(
        ...     model=ChatOpenAI(model="gpt-4.1-mini"),
        ...     tools=[...],
        ...     middleware=[middleware],
        ...     state_schema=PhaseAgentState,
        ...     system_prompt=(
        ...         "You are a research assistant. "
        ...         "You will receive <system> tags with phase guidance "
        ...         "-- follow them."
        ...     ),
        ... )
    """

    state_schema = PhaseAgentState

    def __init__(
        self,
        config: PhaseConfig,
        detector: PhaseDetector | None = None,
    ) -> None:
        """Initialize the phase steering middleware.

        Args:
            config: Phase configuration defining behaviors and transitions.
            detector: Optional custom phase detector. Defaults to
                ``DefaultPhaseDetector`` if not provided.
        """
        super().__init__()
        self.config = config

        from .detector import DefaultPhaseDetector

        self.detector = detector or DefaultPhaseDetector(config)

        initial = config.initial_phase
        if initial is None:
            from .phase import Phase as _Phase

            initial = _Phase.GATHER

        self._current_phase: Phase = initial
        self._phase_history: list[tuple[Phase, int]] = []  # (phase, iteration_at_transition)
        self._phase_iterations: dict[Phase, int] = {}
        self._phase_state: PhaseState = PhaseState()

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Intercept model calls to inject phase guidance and track progression.

        Steps:
        1. Detect current phase from internal state and message history.
        2. Evaluate potential phase transitions.
        3. Apply any validated transition to internal state.
        4. Build and inject phase-specific guidance into the system prompt.
        5. Invoke the model.
        6. Increment phase iteration counters.
        7. Return the response with a state-update Command for observability.

        Args:
            request: Incoming model request containing messages and system prompt.
            handler: Callable that forwards the request to the underlying model.

        Returns:
            ``ExtendedModelResponse`` carrying model output and a ``Command``
            that updates ``current_phase``, ``phase_iteration``, and
            ``total_iterations`` in the agent state.
        """
        messages: list[BaseMessage] = list(request.messages)

        # Detect current phase
        detected_phase = self.detector.detect_phase(self._phase_state, messages)

        # Evaluate and apply transitions
        should_transition, target_phase = self.detector.should_transition(
            detected_phase, self._phase_state, messages
        )

        if should_transition and target_phase is not None and self.config.enable_auto_transition:
            logger.info("Phase transition: %s -> %s", detected_phase.name, target_phase.name)
            self._record_phase_transition(detected_phase, target_phase)
            detected_phase = target_phase
            self._phase_state.record_phase_entry(detected_phase)

        self._current_phase = detected_phase

        # Build phase guidance and inject into message stream
        phase_config = self.config.phases.get(detected_phase)
        if phase_config:
            guidance = self._build_phase_guidance(detected_phase, phase_config)
            steering_msg = SystemMessage(
                content=f"<system>{guidance}</system>"
            )
            modified_messages = list(request.messages) + [steering_msg]
            modified_request = request.override(messages=modified_messages)
            logger.debug(
                "Executing in %s phase (iteration %d)",
                detected_phase.name,
                self._get_phase_iteration(detected_phase) + 1,
            )
        else:
            modified_request = request

        # Call the model
        response = handler(modified_request)

        # Update internal iteration tracking after the call
        self._increment_phase_iteration(detected_phase)
        # Only increment PhaseState if we didn't already via record_phase_entry
        if not (should_transition and target_phase is not None
                and self.config.enable_auto_transition):
            self._phase_state.increment_iteration(detected_phase)

        iteration = self._get_phase_iteration(detected_phase)
        total = self._phase_state.total_iterations

        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "current_phase": detected_phase.name,
                    "phase_iteration": iteration,
                    "total_iterations": total,
                }
            ),
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept tool calls to enforce per-phase allow-lists.

        If the current phase defines a non-empty ``allowed_tools`` list and
        the requested tool is not in it, the call is blocked and a descriptive
        ``ToolMessage`` error is returned without invoking the tool.

        Args:
            request: Incoming tool call request with tool name and arguments.
            handler: Callable that executes the underlying tool.

        Returns:
            ``ToolMessage`` (error or result) or ``Command`` as appropriate.
        """
        phase_config = self.config.phases.get(self._current_phase)
        allowed_tools: list[str] = phase_config.allowed_tools if phase_config else []

        # Empty list means all tools are permitted
        if not allowed_tools:
            return handler(request)

        tool_name = str(request.tool_call.get("name", ""))
        if tool_name not in allowed_tools:
            logger.warning(
                "Tool '%s' blocked in %s phase. Allowed: %s",
                tool_name,
                self._current_phase.name,
                allowed_tools,
            )
            tool_call_id = str(request.tool_call.get("id", "unknown"))
            return ToolMessage(
                content=(
                    f"Tool '{tool_name}' is not available during the "
                    f"{self._current_phase.display_name} phase. "
                    f"Permitted tools in this phase: {', '.join(allowed_tools)}."
                ),
                tool_call_id=tool_call_id,
            )

        return handler(request)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> Phase:
        """Currently active phase."""
        return self._current_phase

    @property
    def phase_history(self) -> list[tuple[Phase, int]]:
        """History of phase transitions with iteration counts at transition time.

        Returns:
            Shallow copy of the internal history list.
        """
        return self._phase_history.copy()

    def reset(self) -> None:
        """Reset middleware state for a new conversation."""
        from .phase import Phase

        initial = (
            self.config.initial_phase if self.config.initial_phase is not None else Phase.GATHER
        )
        self._current_phase = initial
        self._phase_history = []
        self._phase_iterations = {}
        self._phase_state = PhaseState()
        logger.debug("Phase steering middleware reset")

    def should_continue(self, state: PhaseState) -> bool:
        """Check if the workflow should continue or has reached completion.

        Args:
            state: Current workflow state (external PhaseState, not internal).

        Returns:
            ``False`` when in the COMPLETE phase, ``True`` otherwise.
        """
        from .phase import Phase

        return state.current_phase != Phase.COMPLETE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_phase_guidance(self, phase: Phase, config: PhaseBehavior) -> str:
        """Compose the guidance string injected into the system prompt.

        Args:
            phase: Current phase.
            config: Behavior configuration for the phase.

        Returns:
            Formatted multi-line guidance string.
        """
        parts: list[str] = [
            f"[PHASE: {phase.display_name}]",
            config.system_prompt_addition,
        ]

        if config.allowed_tools:
            parts.append(f"Available tools in this phase: {', '.join(config.allowed_tools)}")

        if config.completion_criteria:
            parts.append(f"Phase completion criteria: {config.completion_criteria}")

        return "\n".join(parts)

    def _record_phase_transition(self, from_phase: Phase, to_phase: Phase) -> None:
        """Record a phase transition in history.

        Args:
            from_phase: Phase being left.
            to_phase: Phase being entered.
        """
        self._phase_history.append((from_phase, self._get_phase_iteration(from_phase)))
        logger.info("Transitioned from %s to %s", from_phase.name, to_phase.name)

    def _increment_phase_iteration(self, phase: Phase) -> None:
        """Increment the internal iteration counter for a phase.

        Args:
            phase: Phase whose counter to increment.
        """
        self._phase_iterations[phase] = self._phase_iterations.get(phase, 0) + 1

    def _get_phase_iteration(self, phase: Phase) -> int:
        """Return the current iteration count for a phase.

        Args:
            phase: Phase to query.

        Returns:
            Iteration count (0 if not yet visited).
        """
        return self._phase_iterations.get(phase, 0)
