"""Core middleware implementation for turn-based urgency.

Provides TurnUrgencyMiddleware, which intercepts model calls to inject
urgency-aware guidance and surface tracking fields back into agent state.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.types import Command

from bsm_examples.turn_urgency.calculator import UrgencyCalculator, UrgencyState
from bsm_examples.turn_urgency.state import UrgencyAgentState

if TYPE_CHECKING:
    from bsm_examples.turn_urgency.models import TurnUrgencyConfig

logger = logging.getLogger(__name__)

_TERMINAL_FORCE_LINE = (
    "\n\nYou MUST call the required completion tool on this turn. Do not respond without it."
)


class TurnUrgencyMiddleware(AgentMiddleware):
    """Middleware that dynamically adjusts agent behavior based on remaining turn budget.

    Extends AgentMiddleware and intercepts model calls via wrap_model_call to
    inject urgency-aware guidance into the message stream and surface urgency
    tracking fields back into the agent state via ExtendedModelResponse.

    When ``required_completion_tool`` is configured, the middleware also scans
    model responses for that tool call and forcefully escalates guidance on the
    terminal turn if the tool has not yet been invoked.

    Steering content is injected as a SystemMessage with ``<system>`` tags
    appended to the conversation messages — not by modifying the system prompt.
    The agent's system prompt should mention that ``<system>`` tagged guidance
    will appear in the conversation.

    Example:
        >>> from langchain.agents import create_agent
        >>> config = TurnUrgencyConfig(
        ...     max_turns=5,
        ...     required_completion_tool="recommend_product",
        ... )
        >>> middleware = TurnUrgencyMiddleware(config)
        >>> agent = create_agent(
        ...     model="gpt-4.1-mini",
        ...     tools=[...],
        ...     middleware=[middleware],
        ...     state_schema=UrgencyAgentState,
        ...     system_prompt=(
        ...         "You are a shopping assistant. "
        ...         "You will receive <system> tags with behavioral guidance "
        ...         "— follow them."
        ...     ),
        ... )
    """

    state_schema = UrgencyAgentState

    def __init__(self, config: TurnUrgencyConfig | None = None) -> None:
        """Initialize middleware with configuration.

        Args:
            config: TurnUrgencyConfig instance. Uses defaults if None.
        """
        super().__init__()

        from bsm_examples.turn_urgency.models import TurnUrgencyConfig

        self._config = config or TurnUrgencyConfig()
        self._calculator = UrgencyCalculator(self._config)
        self._last_urgency: UrgencyState | None = None
        self._completion_tool_seen: bool = False

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Intercept model calls to inject urgency guidance and record state.

        Calculates the current urgency level from the conversation history,
        appends an urgency modifier via a SystemMessage, calls the underlying
        model handler, and returns an ExtendedModelResponse that writes urgency
        tracking fields back into the agent state.

        If a required_completion_tool is configured, also scans the response
        for that tool call and escalates guidance on the terminal turn if
        the tool has not yet been seen.

        Args:
            request: Incoming model request with messages and system prompt.
            handler: Callable that executes the actual model call.

        Returns:
            ExtendedModelResponse containing the model response and a Command
            that updates urgency-related fields in the agent state.
        """
        messages = list(request.messages)
        urgency = self._calculator.calculate_from_messages(messages)
        self._log_transition(urgency)

        modified_request = self._build_modified_request(request, urgency)
        response = handler(modified_request)

        # Check if the required completion tool appeared in the response
        if self._config.required_completion_tool and not self._completion_tool_seen:
            self._completion_tool_seen = self._calculator.check_completion_tool(
                list(response.result),
                self._config.required_completion_tool,
            )

        self._last_urgency = urgency

        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "urgency_level": urgency.current_level.name,
                    "turns_used": urgency.turns_used,
                    "turns_remaining": urgency.turns_remaining,
                    "is_terminal": urgency.is_terminal,
                    "completion_tool_called": self._completion_tool_seen,
                }
            ),
        )

    def _log_transition(self, urgency: UrgencyState) -> None:
        """Log urgency level transitions for observability.

        Args:
            urgency: Freshly calculated urgency state.
        """
        previous_level = self._last_urgency.current_level if self._last_urgency else None

        if previous_level != urgency.current_level:
            logger.info(
                "Urgency transition: %s -> %s (turns: %d/%d, remaining: %d)",
                previous_level.name if previous_level else "START",
                urgency.current_level.name,
                urgency.turns_used,
                urgency.max_turns,
                urgency.turns_remaining,
            )

    def _build_modified_request(
        self,
        request: ModelRequest,
        urgency: UrgencyState,
    ) -> ModelRequest:
        """Build a modified ModelRequest with urgency guidance appended to messages.

        Injects a ``<system>``-tagged SystemMessage into the message list. When
        terminal and the required completion tool has not yet been called,
        appends a forceful escalation line to the guidance content.

        Args:
            request: Original model request.
            urgency: Current urgency state.

        Returns:
            A new ModelRequest with an urgency guidance message appended.
        """
        modifier = self._calculator.get_prompt_modifier(urgency)
        guidance = (
            f"<system>URGENCY: {urgency.current_level.name} "
            f"(turn {urgency.turns_used}/{urgency.max_turns})\n"
            f"{modifier}"
        )

        needs_force = (
            urgency.is_terminal
            and self._config.required_completion_tool is not None
            and not self._completion_tool_seen
        )
        if needs_force:
            guidance += _TERMINAL_FORCE_LINE

        guidance += "</system>"

        steering_msg = SystemMessage(content=guidance)
        modified_messages = list(request.messages) + [steering_msg]
        return request.override(messages=modified_messages)

    @property
    def current_urgency(self) -> UrgencyState | None:
        """Get the last calculated urgency state (for inspection/debugging)."""
        return self._last_urgency

    def reset(self) -> None:
        """Reset urgency tracking (call between conversations)."""
        self._last_urgency = None
        self._completion_tool_seen = False
        logger.debug("Turn urgency middleware reset")

    def get_urgency_for_messages(self, messages: Sequence[BaseMessage]) -> UrgencyState:
        """Calculate urgency for a message list without modifying messages.

        Useful for inspection or display purposes.

        Args:
            messages: Message list to analyze.

        Returns:
            UrgencyState for the given messages.
        """
        return self._calculator.calculate_from_messages(messages)
