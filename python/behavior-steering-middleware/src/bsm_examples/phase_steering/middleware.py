"""Phase steering middleware for LangGraph agents.

Injects phase-appropriate guidance into the system prompt and handles
tool-triggered phase transitions via the AgentMiddleware API.
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
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

from bsm_examples.phase_steering.detector import ToolTriggeredDetector
from bsm_examples.phase_steering.models import Phase, PhaseBehavior, PhaseConfig
from bsm_examples.phase_steering.state import PhaseAgentState

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)


class PhaseSteeringMiddleware(AgentMiddleware):
    """Middleware that guides agent execution through tool-triggered workflow phases.

    Implements the Tool-Triggered Phase Steering pattern:

    - ``wrap_model_call``: Injects ``<system>``-tagged phase guidance (current phase
      name, allowed tools, completion criteria) into the system prompt before each
      model call. Returns an ``ExtendedModelResponse`` with phase state updates.

    - ``wrap_tool_call``: Enforces per-phase tool allow-lists before execution, then
      checks for phase transitions AFTER the tool completes. Transitions update
      internal state immediately so the next ``wrap_model_call`` sees the new phase.

    The agent's system prompt should mention that ``<system>`` tags will appear with
    guidance so the model knows to follow them.

    Example:
        >>> middleware = PhaseSteeringMiddleware(config)
        >>> agent = create_agent(
        ...     model=ChatOpenAI(model="gpt-4.1-mini"),
        ...     tools=[browse_catalog, note_preferences, ...],
        ...     middleware=[middleware],
        ...     state_schema=PhaseAgentState,
        ...     system_prompt=(
        ...         "You are a personalization assistant. "
        ...         "Follow <system> tags with phase guidance."
        ...     ),
        ... )
    """

    state_schema = PhaseAgentState

    def __init__(self, config: PhaseConfig) -> None:
        """Initialize the phase steering middleware.

        Args:
            config: Phase configuration defining per-phase behaviors and transition rules.
        """
        super().__init__()
        self.config = config
        self.detector = ToolTriggeredDetector(config)
        self._current_phase: Phase = config.initial_phase
        self._phase_iterations: dict[Phase, int] = {}
        self._phase_history: list[tuple[Phase, int]] = []
        self._total_iterations: int = 0

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Inject phase guidance into the system prompt before each model call.

        Steps:
        1. Look up the ``PhaseBehavior`` for the current phase.
        2. Build a ``<system>``-tagged guidance block with phase name, addition text,
           allowed tools, and completion criteria.
        3. Append a ``SystemMessage`` carrying that guidance to the request messages.
        4. Call the model.
        5. Increment iteration counters.
        6. Return an ``ExtendedModelResponse`` with current phase state updates.

        Args:
            request: Incoming model request with messages and state.
            handler: Callable that forwards the request to the underlying model.

        Returns:
            ``ExtendedModelResponse`` with model output and a ``Command`` updating
            ``current_phase``, ``phase_iteration``, and ``total_iterations``.
        """
        phase_behavior = self.config.phases.get(self._current_phase)

        if phase_behavior:
            guidance = self._build_guidance(self._current_phase, phase_behavior)
            steering_msg = SystemMessage(content=f"<system>{guidance}</system>")
            modified_messages = list(request.messages) + [steering_msg]
            modified_request = request.override(messages=modified_messages)
            logger.debug(
                "Injecting guidance for phase %s (iteration %d)",
                self._current_phase.name,
                self._phase_iterations.get(self._current_phase, 0) + 1,
            )
        else:
            modified_request = request

        response = handler(modified_request)

        # Increment counters after the model responds
        self._phase_iterations[self._current_phase] = (
            self._phase_iterations.get(self._current_phase, 0) + 1
        )
        self._total_iterations += 1

        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "current_phase": self._current_phase.name,
                    "phase_iteration": self._phase_iterations.get(self._current_phase, 0),
                    "total_iterations": self._total_iterations,
                }
            ),
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Enforce tool allow-lists and detect phase transitions after tool execution.

        Steps:
        1. Check the allow-list for the current phase. If the requested tool is not
           permitted, return a ``ToolMessage`` error without calling the tool.
        2. Execute the tool via ``handler``.
        3. Check whether the executed tool triggers a phase transition.
        4. If a transition is found, update internal state and record it in history.

        Args:
            request: Incoming tool call request with tool name and arguments.
            handler: Callable that executes the underlying tool.

        Returns:
            ``ToolMessage`` result or error, or ``Command`` as appropriate.
        """
        tool_name: str = str(request.tool_call["name"] or "")
        tool_call_id: str = str(request.tool_call.get("id") or "unknown")

        phase_behavior = self.config.phases.get(self._current_phase)
        allowed_tools: list[str] = phase_behavior.allowed_tools if phase_behavior else []

        # Enforce allow-list (empty list = all tools permitted)
        if allowed_tools and tool_name not in allowed_tools:
            logger.warning(
                "Tool '%s' blocked in %s phase. Allowed: %s",
                tool_name,
                self._current_phase.name,
                allowed_tools,
            )
            return ToolMessage(
                content=(
                    f"Tool '{tool_name}' is not available during the "
                    f"{self._current_phase.display_name} phase. "
                    f"Permitted tools in this phase: {', '.join(allowed_tools)}."
                ),
                tool_call_id=tool_call_id,
            )

        # Execute the tool
        result = handler(request)

        # Check for phase transition AFTER the tool executes
        next_phase = self.detector.check_tool_transition(self._current_phase, tool_name)
        if next_phase is not None:
            logger.info(
                "Phase transition: %s -> %s (triggered by '%s')",
                self._current_phase.name,
                next_phase.name,
                tool_name,
            )
            self._phase_history.append(
                (self._current_phase, self._phase_iterations.get(self._current_phase, 0))
            )
            self._current_phase = next_phase

        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> Phase:
        """Currently active phase."""
        return self._current_phase

    @property
    def phase_history(self) -> list[tuple[Phase, int]]:
        """History of phase transitions as (phase_left, iteration_count_at_transition) pairs."""
        return self._phase_history.copy()

    def reset(self) -> None:
        """Reset all middleware state for a new conversation."""
        self._current_phase = self.config.initial_phase
        self._phase_iterations = {}
        self._phase_history = []
        self._total_iterations = 0
        logger.debug("PhaseSteeringMiddleware reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_guidance(self, phase: Phase, behavior: PhaseBehavior) -> str:
        """Compose the guidance text injected as a ``<system>`` message.

        Args:
            phase: The current phase (used for the phase label).
            behavior: Behavior configuration for the phase.

        Returns:
            Multi-line guidance string (without the outer ``<system>`` tags).
        """
        parts: list[str] = [
            f"[PHASE: {phase.display_name}]",
            behavior.system_prompt_addition,
        ]

        if behavior.allowed_tools:
            parts.append(f"Available tools: {', '.join(behavior.allowed_tools)}")

        if behavior.completion_criteria:
            parts.append(f"Completion criteria: {behavior.completion_criteria}")

        return "\n".join(parts)
