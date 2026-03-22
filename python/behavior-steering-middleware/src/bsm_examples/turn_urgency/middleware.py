"""Core middleware implementation for turn-based urgency.

This module contains the UrgencyCalculator and TurnUrgencyMiddleware classes
that implement the behavior-steering pattern using the AgentMiddleware API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from bsm_examples.turn_urgency.types import TurnUrgencyConfig

from bsm_examples.turn_urgency.types import UrgencyAgentState, UrgencyLevel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UrgencyState:
    """Immutable snapshot of current urgency calculation."""

    current_level: UrgencyLevel
    turns_used: int
    turns_remaining: int
    max_turns: int
    remaining_ratio: float
    is_terminal: bool


class UrgencyCalculator:
    """Calculates urgency levels based on turn consumption.

    This class is stateless and thread-safe. It performs calculations
    based on provided message history and configuration.

    Example:
        >>> config = TurnUrgencyConfig(max_turns=5)
        >>> calculator = UrgencyCalculator(config)
        >>> state = calculator.calculate_from_messages(messages)
        >>> print(state.current_level)
        UrgencyLevel.EXPLORATION
    """

    def __init__(self, config: TurnUrgencyConfig) -> None:
        """Initialize calculator with configuration.

        Args:
            config: TurnUrgencyConfig with thresholds and prompts.
        """
        self._config = config

    def calculate_from_messages(
        self,
        messages: list[BaseMessage],
    ) -> UrgencyState:
        """Calculate urgency state from message history.

        Args:
            messages: Full conversation history.

        Returns:
            UrgencyState with current level and metrics.
        """
        turns_used = self._count_turns(messages)
        return self._calculate(turns_used)

    def _count_turns(self, messages: list[BaseMessage]) -> int:
        """Count turns based on configuration.

        Supports three counting modes:
        - Default: Count conversation pairs (max of human/agent messages)
        - Human-only: Count only human messages
        - Agent-only: Count only agent messages

        Args:
            messages: List of messages to count.

        Returns:
            Number of turns used.
        """
        if self._config.count_human_turns_only:
            return sum(1 for m in messages if isinstance(m, HumanMessage))
        elif self._config.count_agent_turns_only:
            return sum(1 for m in messages if isinstance(m, AIMessage))
        else:
            # Count conversation pairs (human + agent = 1 turn)
            # A turn is complete when both sides have spoken
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            agent_count = sum(1 for m in messages if isinstance(m, AIMessage))
            return max(human_count, agent_count)

    def _calculate(self, turns_used: int) -> UrgencyState:
        """Calculate urgency state from turn count.

        Args:
            turns_used: Number of turns already consumed.

        Returns:
            UrgencyState with calculated level and metrics.
        """
        max_turns = self._config.max_turns
        turns_remaining = max(0, max_turns - turns_used)
        remaining_ratio = turns_remaining / max_turns if max_turns > 0 else 0.0

        level = self._determine_level(remaining_ratio)

        return UrgencyState(
            current_level=level,
            turns_used=turns_used,
            turns_remaining=turns_remaining,
            max_turns=max_turns,
            remaining_ratio=remaining_ratio,
            is_terminal=level == UrgencyLevel.TERMINAL,
        )

    def _determine_level(self, remaining_ratio: float) -> UrgencyLevel:
        """Map remaining ratio to urgency level.

        Args:
            remaining_ratio: Ratio of turns remaining (0.0 to 1.0).

        Returns:
            UrgencyLevel based on configured thresholds.
        """
        thresholds = self._config.thresholds

        if remaining_ratio > thresholds.balanced_max:
            return UrgencyLevel.EXPLORATION
        elif remaining_ratio > thresholds.efficiency_max:
            return UrgencyLevel.BALANCED
        elif remaining_ratio > 0:
            return UrgencyLevel.EFFICIENCY
        else:
            return UrgencyLevel.TERMINAL

    def get_prompt_modifier(self, state: UrgencyState) -> str:
        """Get the appropriate prompt modifier for current urgency level.

        Args:
            state: Current urgency state.

        Returns:
            Prompt modifier string for the current urgency level.
        """
        prompts = self._config.prompts

        match state.current_level:
            case UrgencyLevel.EXPLORATION:
                return prompts.exploration
            case UrgencyLevel.BALANCED:
                return prompts.balanced
            case UrgencyLevel.EFFICIENCY:
                return prompts.efficiency
            case UrgencyLevel.TERMINAL:
                return prompts.terminal


class TurnUrgencyMiddleware(AgentMiddleware):
    """Middleware that dynamically adjusts agent behavior based on remaining turn budget.

    Extends AgentMiddleware and intercepts model calls via wrap_model_call to
    inject urgency-aware guidance into the message stream and surface urgency
    tracking fields back into the agent state via ExtendedModelResponse.

    Steering content is injected as a SystemMessage with ``<system>`` tags
    appended to the conversation messages — not by modifying the system prompt.
    The agent's system prompt should mention that ``<system>`` tagged guidance
    will appear in the conversation.

    Example:
        >>> from langchain.agents import create_agent
        >>> config = TurnUrgencyConfig(max_turns=5)
        >>> middleware = TurnUrgencyMiddleware(config)
        >>> agent = create_agent(
        ...     model="gpt-4.1-mini",
        ...     tools=[...],
        ...     middleware=[middleware],
        ...     state_schema=UrgencyAgentState,
        ...     system_prompt=(
        ...         "You are a helpful assistant. "
        ...         "You will receive <system> tags with behavioral guidance "
        ...         "— follow them."
        ...     ),
        ... )

    The middleware automatically:
    1. Counts turns from the model request's message history
    2. Calculates the current urgency level
    3. Appends a ``<system>``-tagged message with urgency guidance
    4. Logs urgency transitions for observability
    5. Returns updated urgency fields in the agent state
    """

    state_schema = UrgencyAgentState

    def __init__(self, config: TurnUrgencyConfig | None = None) -> None:
        """Initialize middleware with configuration.

        Args:
            config: TurnUrgencyConfig instance. Uses defaults if None.
        """
        super().__init__()

        from bsm_examples.turn_urgency.types import TurnUrgencyConfig

        self._config = config or TurnUrgencyConfig()
        self._calculator = UrgencyCalculator(self._config)
        self._last_urgency: UrgencyState | None = None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Intercept model calls to inject urgency guidance and record state.

        Calculates the current urgency level from the conversation history,
        appends an urgency modifier to the system prompt, calls the underlying
        model handler, and returns an ExtendedModelResponse that writes urgency
        tracking fields back into the agent state.

        Args:
            request: Incoming model request with messages and system prompt.
            handler: Callable that executes the actual model call.

        Returns:
            ExtendedModelResponse containing the model response and a Command
            that updates urgency-related fields in the agent state.
        """
        urgency = self._calculator.calculate_from_messages(list(request.messages))
        self._log_transition(urgency)

        modified_request = self._build_modified_request(request, urgency)
        response = handler(modified_request)

        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "urgency_level": urgency.current_level.name,
                    "turns_used": urgency.turns_used,
                    "turns_remaining": urgency.turns_remaining,
                    "is_terminal": urgency.is_terminal,
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

        self._last_urgency = urgency

    def _build_modified_request(
        self,
        request: ModelRequest,
        urgency: UrgencyState,
    ) -> ModelRequest:
        """Build a modified ModelRequest with urgency guidance appended to messages.

        Injects a ``<system>``-tagged SystemMessage into the message list rather
        than modifying the system prompt. This keeps the system prompt static and
        allows steering content to flow naturally in the conversation.

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
            f"{modifier}</system>"
        )
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
        logger.debug("Turn urgency middleware reset")

    def get_urgency_for_messages(self, messages: list[BaseMessage]) -> UrgencyState:
        """Calculate urgency for a message list without modifying messages.

        Useful for inspection or display purposes.

        Args:
            messages: Message list to analyze.

        Returns:
            UrgencyState for the given messages.
        """
        return self._calculator.calculate_from_messages(messages)
