"""Unit tests for turn-based urgency middleware.

Tests cover:
- UrgencyLevel enum behavior
- UrgencyThresholds validation
- UrgencyPrompts defaults
- TurnUrgencyConfig validation
- UrgencyCalculator turn counting
- UrgencyCalculator level determination
- TurnUrgencyMiddleware message modification
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bsm_examples.turn_urgency import (
    TurnUrgencyConfig,
    TurnUrgencyMiddleware,
    UrgencyCalculator,
    UrgencyLevel,
    UrgencyPrompts,
    UrgencyThresholds,
)


class TestUrgencyLevel:
    """Tests for UrgencyLevel enum."""

    def test_enum_values_exist(self) -> None:
        """All four urgency levels should exist."""
        assert UrgencyLevel.EXPLORATION is not None
        assert UrgencyLevel.BALANCED is not None
        assert UrgencyLevel.EFFICIENCY is not None
        assert UrgencyLevel.TERMINAL is not None

    def test_enum_ordering(self) -> None:
        """Enum values should have consistent auto values."""
        # Auto values are assigned sequentially
        levels = list(UrgencyLevel)
        assert len(levels) == 4
        assert levels[0] == UrgencyLevel.EXPLORATION
        assert levels[1] == UrgencyLevel.BALANCED
        assert levels[2] == UrgencyLevel.EFFICIENCY
        assert levels[3] == UrgencyLevel.TERMINAL


class TestUrgencyThresholds:
    """Tests for UrgencyThresholds model."""

    def test_default_thresholds(self) -> None:
        """Default thresholds should match design spec."""
        thresholds = UrgencyThresholds()
        assert thresholds.exploration_max == 1.0
        assert thresholds.balanced_max == 0.6
        assert thresholds.efficiency_max == 0.3

    def test_valid_custom_thresholds(self) -> None:
        """Custom thresholds should be accepted if valid."""
        thresholds = UrgencyThresholds(
            exploration_max=1.0,
            balanced_max=0.5,
            efficiency_max=0.2,
        )
        assert thresholds.balanced_max == 0.5
        assert thresholds.efficiency_max == 0.2

    def test_invalid_threshold_order(self) -> None:
        """Thresholds must be in descending order."""
        with pytest.raises(ValueError, match="balanced_max must be less than exploration_max"):
            UrgencyThresholds(exploration_max=0.6, balanced_max=0.8)

        with pytest.raises(ValueError, match="efficiency_max must be less than balanced_max"):
            UrgencyThresholds(balanced_max=0.3, efficiency_max=0.5)

    def test_threshold_bounds(self) -> None:
        """Thresholds must be between 0 and 1."""
        with pytest.raises(ValueError):
            UrgencyThresholds(exploration_max=1.5)

        with pytest.raises(ValueError):
            UrgencyThresholds(balanced_max=-0.1)


class TestUrgencyPrompts:
    """Tests for UrgencyPrompts model."""

    def test_default_prompts_exist(self) -> None:
        """Default prompts should be non-empty strings."""
        prompts = UrgencyPrompts()
        assert len(prompts.exploration) > 0
        assert len(prompts.balanced) > 0
        assert len(prompts.efficiency) > 0
        assert len(prompts.terminal) > 0

    def test_custom_prompts(self) -> None:
        """Custom prompts should be accepted."""
        custom_exploration = "Custom exploration prompt"
        prompts = UrgencyPrompts(exploration=custom_exploration)
        assert prompts.exploration == custom_exploration
        # Other prompts should still have defaults
        assert len(prompts.balanced) > 0

    def test_terminal_prompt_contains_final(self) -> None:
        """Terminal prompt should indicate final turn."""
        prompts = UrgencyPrompts()
        assert "FINAL" in prompts.terminal.upper()


class TestTurnUrgencyConfig:
    """Tests for TurnUrgencyConfig model."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = TurnUrgencyConfig()
        assert config.max_turns == 5
        assert config.count_human_turns_only is False
        assert config.count_agent_turns_only is False

    def test_custom_max_turns(self) -> None:
        """Max turns should be configurable within bounds."""
        config = TurnUrgencyConfig(max_turns=10)
        assert config.max_turns == 10

    def test_max_turns_bounds(self) -> None:
        """Max turns must be between 1 and 50."""
        with pytest.raises(ValueError):
            TurnUrgencyConfig(max_turns=0)

        with pytest.raises(ValueError):
            TurnUrgencyConfig(max_turns=51)

    def test_human_only_counting(self) -> None:
        """Human-only counting mode should be configurable."""
        config = TurnUrgencyConfig(count_human_turns_only=True)
        assert config.count_human_turns_only is True
        assert config.count_agent_turns_only is False

    def test_agent_only_counting(self) -> None:
        """Agent-only counting mode should be configurable."""
        config = TurnUrgencyConfig(count_agent_turns_only=True)
        assert config.count_agent_turns_only is True
        assert config.count_human_turns_only is False

    def test_both_counting_modes_invalid(self) -> None:
        """Cannot enable both counting modes simultaneously."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            TurnUrgencyConfig(
                count_human_turns_only=True,
                count_agent_turns_only=True,
            )


class TestUrgencyCalculator:
    """Tests for UrgencyCalculator."""

    @pytest.fixture
    def default_config(self) -> TurnUrgencyConfig:
        """Default configuration fixture."""
        return TurnUrgencyConfig(max_turns=5)

    @pytest.fixture
    def calculator(self, default_config: TurnUrgencyConfig) -> UrgencyCalculator:
        """Calculator fixture with default config."""
        return UrgencyCalculator(default_config)

    def test_calculator_initialization(self, calculator: UrgencyCalculator) -> None:
        """Calculator should initialize with config."""
        assert calculator is not None

    def test_count_turns_default_mode(
        self, calculator: UrgencyCalculator
    ) -> None:
        """Default mode counts max of human/agent messages."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="How are you?"),
        ]
        # 2 human, 1 agent -> max = 2
        assert calculator._count_turns(messages) == 2

    def test_count_turns_human_only(
        self, default_config: TurnUrgencyConfig
    ) -> None:
        """Human-only mode counts only human messages."""
        default_config.count_human_turns_only = True
        calculator = UrgencyCalculator(default_config)

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm fine"),
        ]
        # Should count 2 human messages
        assert calculator._count_turns(messages) == 2

    def test_count_turns_agent_only(
        self, default_config: TurnUrgencyConfig
    ) -> None:
        """Agent-only mode counts only agent messages."""
        default_config.count_agent_turns_only = True
        calculator = UrgencyCalculator(default_config)

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm fine"),
        ]
        # Should count 2 agent messages
        assert calculator._count_turns(messages) == 2

    def test_empty_messages(self, calculator: UrgencyCalculator) -> None:
        """Empty message list should result in 0 turns used."""
        state = calculator.calculate_from_messages([])
        assert state.turns_used == 0
        assert state.turns_remaining == 5
        assert state.current_level == UrgencyLevel.EXPLORATION

    def test_exploration_level(self, calculator: UrgencyCalculator) -> None:
        """High remaining ratio should yield EXPLORATION."""
        # 4 turns remaining out of 5 = 80% -> EXPLORATION
        messages = [HumanMessage(content="Hello")]
        state = calculator.calculate_from_messages(messages)
        assert state.current_level == UrgencyLevel.EXPLORATION
        assert state.remaining_ratio == 0.8

    def test_balanced_level(self, calculator: UrgencyCalculator) -> None:
        """Medium remaining ratio should yield BALANCED."""
        # 3 turns remaining out of 5 = 60% -> BALANCED (at threshold)
        # Actually > 0.6 is EXPLORATION, so we need 50% = 2.5 turns used
        # Let's use 3 turns used = 40% remaining -> EFFICIENCY
        # 2 turns used = 60% remaining -> BALANCED
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            HumanMessage(content="Question"),
        ]
        state = calculator.calculate_from_messages(messages)
        # 3 messages, max = 2 human or 1 agent -> 2 turns used
        # 3 remaining out of 5 = 60% -> BALANCED
        assert state.turns_used == 2
        assert state.remaining_ratio == 0.6
        assert state.current_level == UrgencyLevel.BALANCED

    def test_efficiency_level(self, calculator: UrgencyCalculator) -> None:
        """Low remaining ratio should yield EFFICIENCY."""
        # 4 turns used, 1 remaining = 20% -> EFFICIENCY
        messages = [
            HumanMessage(content="1"),
            AIMessage(content="1"),
            HumanMessage(content="2"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="3"),
            HumanMessage(content="4"),
            AIMessage(content="4"),
        ]
        state = calculator.calculate_from_messages(messages)
        assert state.turns_used == 4
        assert state.remaining_ratio == 0.2
        assert state.current_level == UrgencyLevel.EFFICIENCY

    def test_terminal_level(self, calculator: UrgencyCalculator) -> None:
        """Zero remaining turns should yield TERMINAL."""
        # 5+ turns used -> 0 remaining -> TERMINAL
        messages = [
            HumanMessage(content="1"),
            AIMessage(content="1"),
            HumanMessage(content="2"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
            AIMessage(content="3"),
            HumanMessage(content="4"),
            AIMessage(content="4"),
            HumanMessage(content="5"),
            AIMessage(content="5"),
        ]
        state = calculator.calculate_from_messages(messages)
        assert state.turns_used == 5
        assert state.remaining_ratio == 0.0
        assert state.current_level == UrgencyLevel.TERMINAL
        assert state.is_terminal is True

    def test_get_prompt_modifier_exploration(
        self, calculator: UrgencyCalculator
    ) -> None:
        """Should return exploration prompt for EXPLORATION level."""
        from bsm_examples.turn_urgency.middleware import UrgencyState

        state = UrgencyState(
            current_level=UrgencyLevel.EXPLORATION,
            turns_used=0,
            turns_remaining=5,
            max_turns=5,
            remaining_ratio=1.0,
            is_terminal=False,
        )
        modifier = calculator.get_prompt_modifier(state)
        assert "ample turns" in modifier.lower()

    def test_get_prompt_modifier_terminal(
        self, calculator: UrgencyCalculator
    ) -> None:
        """Should return terminal prompt for TERMINAL level."""
        from bsm_examples.turn_urgency.middleware import UrgencyState

        state = UrgencyState(
            current_level=UrgencyLevel.TERMINAL,
            turns_used=5,
            turns_remaining=0,
            max_turns=5,
            remaining_ratio=0.0,
            is_terminal=True,
        )
        modifier = calculator.get_prompt_modifier(state)
        assert "FINAL" in modifier


class TestTurnUrgencyMiddleware:
    """Tests for TurnUrgencyMiddleware."""

    @pytest.fixture
    def middleware(self) -> TurnUrgencyMiddleware:
        """Middleware fixture with default config."""
        return TurnUrgencyMiddleware(TurnUrgencyConfig(max_turns=5))

    def test_middleware_initialization(self, middleware: TurnUrgencyMiddleware) -> None:
        """Middleware should initialize with config."""
        assert middleware is not None
        assert middleware.current_urgency is None

    def test_process_messages_injects_urgency(
        self, middleware: TurnUrgencyMiddleware
    ) -> None:
        """Middleware should inject urgency into system prompt."""
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
        ]
        modified, urgency = middleware.process_messages(messages)

        # Should have same number of messages
        assert len(modified) == 2

        # System message should contain urgency marker
        system_msg = modified[0]
        assert isinstance(system_msg, SystemMessage)
        assert "[URGENCY:" in system_msg.content
        assert urgency.current_level == UrgencyLevel.EXPLORATION

    def test_process_messages_creates_system_if_missing(
        self, middleware: TurnUrgencyMiddleware
    ) -> None:
        """Middleware should create system message if none exists."""
        messages = [HumanMessage(content="Hello")]
        modified, urgency = middleware.process_messages(messages)

        # Should now have 2 messages (system + human)
        assert len(modified) == 2
        assert isinstance(modified[0], SystemMessage)
        assert "[URGENCY:" in modified[0].content

    def test_urgency_transition_tracking(
        self, middleware: TurnUrgencyMiddleware
    ) -> None:
        """Middleware should track urgency transitions."""
        # First call - EXPLORATION
        messages1 = [HumanMessage(content="Hello")]
        _, urgency1 = middleware.process_messages(messages1)
        assert urgency1.current_level == UrgencyLevel.EXPLORATION
        assert middleware.current_urgency == urgency1

        # Second call with more messages - should transition
        messages2 = [
            HumanMessage(content="1"),
            AIMessage(content="1"),
            HumanMessage(content="2"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
        ]
        _, urgency2 = middleware.process_messages(messages2)
        assert urgency2.current_level == UrgencyLevel.BALANCED

    def test_reset_clears_urgency(self, middleware: TurnUrgencyMiddleware) -> None:
        """Reset should clear urgency tracking."""
        messages = [HumanMessage(content="Hello")]
        middleware.process_messages(messages)
        assert middleware.current_urgency is not None

        middleware.reset()
        assert middleware.current_urgency is None

    def test_get_urgency_for_messages(self, middleware: TurnUrgencyMiddleware) -> None:
        """Should calculate urgency without modifying messages."""
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
        ]
        urgency = middleware.get_urgency_for_messages(messages)

        assert urgency.current_level == UrgencyLevel.EXPLORATION
        # Original messages should be unchanged
        assert messages[0].content == "You are helpful."


class TestIntegration:
    """Integration tests for the complete middleware flow."""

    def test_full_conversation_flow(self) -> None:
        """Test a complete conversation with urgency escalation."""
        config = TurnUrgencyConfig(max_turns=3)
        middleware = TurnUrgencyMiddleware(config)

        # Turn 1: EXPLORATION (1 turn used, 2 remaining = 67%)
        messages1 = [HumanMessage(content="Hello")]
        modified1, urgency1 = middleware.process_messages(messages1)
        assert urgency1.current_level == UrgencyLevel.EXPLORATION
        messages1.append(AIMessage(content="Hi"))

        # Turn 2: BALANCED (2 turns used, 1 remaining = 33%)
        # 33% > 30% (efficiency_max), so it's BALANCED
        messages2 = messages1 + [HumanMessage(content="Question")]
        modified2, urgency2 = middleware.process_messages(messages2)
        assert urgency2.turns_used == 2
        assert urgency2.remaining_ratio == 1 / 3  # ~33%
        assert urgency2.current_level == UrgencyLevel.BALANCED

        # Turn 3: TERMINAL (3 turns used, 0 remaining = 0%)
        messages3 = messages2 + [AIMessage(content="Answer"), HumanMessage(content="Thanks")]
        modified3, urgency3 = middleware.process_messages(messages3)
        # 3 human, 2 agent = 3 turns used
        # 0 remaining = 0% -> TERMINAL
        assert urgency3.current_level == UrgencyLevel.TERMINAL
        assert urgency3.is_terminal is True

    def test_custom_thresholds_flow(self) -> None:
        """Test with custom urgency thresholds."""
        config = TurnUrgencyConfig(
            max_turns=10,
            thresholds=UrgencyThresholds(
                exploration_max=1.0,
                balanced_max=0.5,
                efficiency_max=0.2,
            ),
        )
        middleware = TurnUrgencyMiddleware(config)

        # 6 turns used, 4 remaining = 40% -> BALANCED (between 20% and 50%)
        messages = []
        for i in range(6):
            messages.extend([HumanMessage(content=f"{i}"), AIMessage(content=f"{i}")])

        _, urgency = middleware.process_messages(messages)
        assert urgency.turns_used == 6
        assert urgency.remaining_ratio == 0.4
        assert urgency.current_level == UrgencyLevel.BALANCED

        # Add more messages to reach EFFICIENCY
        # 8 turns used, 2 remaining = 20% -> EFFICIENCY (above 0%, below 20%)
        messages.extend([
            HumanMessage(content="x"), AIMessage(content="x"),
            HumanMessage(content="y"), AIMessage(content="y"),
        ])
        _, urgency2 = middleware.process_messages(messages)
        assert urgency2.turns_used == 8
        assert urgency2.remaining_ratio == 0.2
        # At exactly 20%, it's EFFICIENCY (not > 20%)
        assert urgency2.current_level == UrgencyLevel.EFFICIENCY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
