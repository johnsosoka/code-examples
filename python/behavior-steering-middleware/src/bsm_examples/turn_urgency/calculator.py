"""Urgency calculation logic for turn-based urgency middleware.

Provides the UrgencyState dataclass and UrgencyCalculator class that
compute urgency levels from message history. Both are stateless and
thread-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from bsm_examples.turn_urgency.models import UrgencyLevel

if TYPE_CHECKING:
    from bsm_examples.turn_urgency.models import TurnUrgencyConfig


@dataclass(frozen=True)
class UrgencyState:
    """Immutable snapshot of current urgency calculation."""

    current_level: UrgencyLevel
    turns_used: int
    turns_remaining: int
    max_turns: int
    remaining_ratio: float
    is_terminal: bool
    completion_tool_called: bool = False


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
        messages: Sequence[BaseMessage],
    ) -> UrgencyState:
        """Calculate urgency state from message history.

        Args:
            messages: Full conversation history.

        Returns:
            UrgencyState with current level and metrics.
        """
        turns_used = self._count_turns(messages)
        return self._calculate(turns_used)

    def _count_turns(self, messages: Sequence[BaseMessage]) -> int:
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

    def check_completion_tool(
        self,
        messages: Sequence[BaseMessage],
        tool_name: str,
    ) -> bool:
        """Scan message history to check if the required completion tool was called.

        Inspects all AIMessage tool_calls across the conversation for a matching
        tool name. Returns True as soon as a match is found.

        Args:
            messages: Full conversation history to scan.
            tool_name: Name of the tool to look for.

        Returns:
            True if the tool was called at least once, False otherwise.
        """
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            tool_calls = getattr(message, "tool_calls", None) or []
            for call in tool_calls:
                # tool_calls entries are dicts with a "name" key
                if isinstance(call, dict):
                    call_name = call.get("name")
                else:
                    call_name = getattr(call, "name", None)
                if call_name == tool_name:
                    return True
        return False
