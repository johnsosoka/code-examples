"""Core middleware implementation for turn-based urgency.

This module contains the UrgencyCalculator and TurnUrgencyMiddleware classes
that implement the behavior-steering pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from bsm_examples.turn_urgency.types import TurnUrgencyConfig

from bsm_examples.turn_urgency.types import UrgencyLevel

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

        # Determine urgency level based on thresholds
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


class TurnUrgencyMiddleware:
    """Middleware that dynamically adjusts agent behavior based on remaining turn budget.

    This middleware intercepts model calls and modifies the system prompt to reflect
    current urgency level, creating progressive behavior escalation as turns are consumed.

    Example:
        >>> config = TurnUrgencyConfig(max_turns=5)
        >>> middleware = TurnUrgencyMiddleware(config)
        >>>
        >>> # Use with LangGraph
        >>> from langgraph.graph import StateGraph
        >>> builder = StateGraph(TurnUrgencyState)
        >>> # ... add nodes that use the middleware

    The middleware automatically:
    1. Counts turns from message history
    2. Calculates current urgency level
    3. Injects urgency-appropriate instructions into system prompt
    4. Logs urgency transitions for observability
    """

    def __init__(self, config: TurnUrgencyConfig | None = None) -> None:
        """Initialize middleware with configuration.

        Args:
            config: TurnUrgencyConfig instance. Uses defaults if None.
        """
        from bsm_examples.turn_urgency.types import TurnUrgencyConfig

        self._config = config or TurnUrgencyConfig()
        self._calculator = UrgencyCalculator(self._config)
        self._last_urgency: UrgencyState | None = None

    def process_messages(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], UrgencyState]:
        """Process messages and inject urgency-aware system prompt.

        This is the main entry point for using the middleware without
        the full LangGraph integration.

        Args:
            messages: Current conversation messages.

        Returns:
            Tuple of (modified messages, urgency state).
        """
        # Calculate current urgency from message history
        urgency = self._calculator.calculate_from_messages(messages)

        # Log urgency transitions
        if self._last_urgency is None or self._last_urgency.current_level != urgency.current_level:
            logger.info(
                "Urgency transition: %s -> %s (turns: %d/%d, remaining: %d)",
                self._last_urgency.current_level.name if self._last_urgency else "START",
                urgency.current_level.name,
                urgency.turns_used,
                urgency.max_turns,
                urgency.turns_remaining,
            )
            self._last_urgency = urgency

        # Modify messages with urgency-aware system prompt
        modified_messages = self._inject_urgency_prompt(messages, urgency)

        return modified_messages, urgency

    def _inject_urgency_prompt(
        self,
        messages: list[BaseMessage],
        urgency: UrgencyState,
    ) -> list[BaseMessage]:
        """Inject urgency modifier into system prompt.

        Strategy: Find existing system message and append urgency modifier,
        or create new system message if none exists.

        Args:
            messages: Original message list.
            urgency: Current urgency state.

        Returns:
            Modified message list with urgency prompt injected.
        """
        urgency_modifier = self._calculator.get_prompt_modifier(urgency)
        modified_messages: list[BaseMessage] = []
        system_modified = False

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Append urgency modifier to existing system prompt
                modified_content = (
                    f"{msg.content}\n\n"
                    f"[URGENCY: {urgency.current_level.name}] {urgency_modifier}"
                )
                modified_messages.append(SystemMessage(content=modified_content))
                system_modified = True
            else:
                modified_messages.append(msg)

        if not system_modified:
            # Prepend urgency system message
            urgency_content = f"[URGENCY: {urgency.current_level.name}] {urgency_modifier}"
            modified_messages.insert(0, SystemMessage(content=urgency_content))

        return modified_messages

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
