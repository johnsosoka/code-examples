"""Unit tests for turn-based urgency middleware.

Tests cover:
- UrgencyLevel enum behavior
- UrgencyThresholds validation
- UrgencyPrompts defaults (shopping-themed)
- TurnUrgencyConfig validation including required_completion_tool
- UrgencyCalculator turn counting and level determination
- UrgencyCalculator.check_completion_tool
- TurnUrgencyMiddleware wrap_model_call integration
- Completion tool tracking and terminal escalation
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
    UrgencyState,
    UrgencyThresholds,
)
from tests.helpers import make_mock_handler, make_mock_request


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
    """Tests for UrgencyPrompts model with shopping-themed defaults."""

    def test_default_prompts_exist(self) -> None:
        """Default prompts should be non-empty strings."""
        prompts = UrgencyPrompts()
        assert len(prompts.exploration) > 0
        assert len(prompts.balanced) > 0
        assert len(prompts.efficiency) > 0
        assert len(prompts.terminal) > 0

    def test_default_exploration_prompt_is_shopping_themed(self) -> None:
        """Default exploration prompt should reference customer preferences."""
        prompts = UrgencyPrompts()
        lower = prompts.exploration.lower()
        assert "customer" in lower or "preferences" in lower

    def test_default_balanced_prompt_narrows_options(self) -> None:
        """Default balanced prompt should indicate narrowing down."""
        prompts = UrgencyPrompts()
        assert "narrow" in prompts.balanced.lower() or "narrowing" in prompts.balanced.lower()

    def test_default_efficiency_prompt_references_recommend_tool(self) -> None:
        """Default efficiency prompt should mention the recommend_product tool."""
        prompts = UrgencyPrompts()
        assert "recommend_product" in prompts.efficiency

    def test_terminal_prompt_contains_final(self) -> None:
        """Terminal prompt should indicate final turn."""
        prompts = UrgencyPrompts()
        assert "FINAL" in prompts.terminal.upper()

    def test_terminal_prompt_references_recommend_tool(self) -> None:
        """Terminal prompt should explicitly call for recommend_product tool use."""
        prompts = UrgencyPrompts()
        assert "recommend_product" in prompts.terminal

    def test_custom_prompts(self) -> None:
        """Custom prompts should be accepted."""
        custom_exploration = "Custom exploration prompt"
        prompts = UrgencyPrompts(exploration=custom_exploration)
        assert prompts.exploration == custom_exploration
        # Other prompts should still have defaults
        assert len(prompts.balanced) > 0


class TestTurnUrgencyConfig:
    """Tests for TurnUrgencyConfig model."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = TurnUrgencyConfig()
        assert config.max_turns == 5
        assert config.count_human_turns_only is False
        assert config.count_agent_turns_only is False
        assert config.required_completion_tool is None

    def test_required_completion_tool_configurable(self) -> None:
        """required_completion_tool should accept a tool name string."""
        config = TurnUrgencyConfig(required_completion_tool="recommend_product")
        assert config.required_completion_tool == "recommend_product"

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

    def test_count_turns_default_mode(self, calculator: UrgencyCalculator) -> None:
        """Default mode counts max of human/agent messages."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="How are you?"),
        ]
        # 2 human, 1 agent -> max = 2
        assert calculator._count_turns(messages) == 2

    def test_count_turns_human_only(self, default_config: TurnUrgencyConfig) -> None:
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

    def test_count_turns_agent_only(self, default_config: TurnUrgencyConfig) -> None:
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
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            HumanMessage(content="Question"),
        ]
        state = calculator.calculate_from_messages(messages)
        # 3 messages, max(2 human, 1 agent) = 2 turns used
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

    def test_get_prompt_modifier_exploration(self, calculator: UrgencyCalculator) -> None:
        """Should return exploration prompt for EXPLORATION level."""
        state = UrgencyState(
            current_level=UrgencyLevel.EXPLORATION,
            turns_used=0,
            turns_remaining=5,
            max_turns=5,
            remaining_ratio=1.0,
            is_terminal=False,
        )
        modifier = calculator.get_prompt_modifier(state)
        # Shopping-themed default: mentions customer preferences
        assert len(modifier) > 0
        assert "customer" in modifier.lower() or "preferences" in modifier.lower()

    def test_get_prompt_modifier_terminal(self, calculator: UrgencyCalculator) -> None:
        """Should return terminal prompt for TERMINAL level."""
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

    def test_check_completion_tool_found(self, calculator: UrgencyCalculator) -> None:
        """Should return True when the target tool appears in AIMessage tool_calls."""
        recommend_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "recommend_product",
                    "id": "call_1",
                    "args": {"product_name": "X", "reason": "Y"},
                },
            ],
        )
        messages = [
            HumanMessage(content="Help me find headphones"),
            recommend_call,
        ]
        assert calculator.check_completion_tool(messages, "recommend_product") is True

    def test_check_completion_tool_not_found(self, calculator: UrgencyCalculator) -> None:
        """Should return False when the target tool has not been called."""
        messages = [
            HumanMessage(content="Help me find headphones"),
            AIMessage(content="Sure, let me search."),
        ]
        assert calculator.check_completion_tool(messages, "recommend_product") is False

    def test_check_completion_tool_wrong_tool(self, calculator: UrgencyCalculator) -> None:
        """Should return False when a different tool was called."""
        search_call = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_products", "id": "call_2", "args": {"query": "headphones"}},
            ],
        )
        messages = [HumanMessage(content="Find headphones"), search_call]
        assert calculator.check_completion_tool(messages, "recommend_product") is False

    def test_check_completion_tool_ignores_human_messages(
        self, calculator: UrgencyCalculator
    ) -> None:
        """Should only scan AIMessage instances, not HumanMessages."""
        messages = [HumanMessage(content="Please call recommend_product")]
        assert calculator.check_completion_tool(messages, "recommend_product") is False


class TestTurnUrgencyMiddleware:
    """Tests for TurnUrgencyMiddleware."""

    @pytest.fixture
    def middleware(self) -> TurnUrgencyMiddleware:
        """Middleware fixture with default config."""
        return TurnUrgencyMiddleware(TurnUrgencyConfig(max_turns=5))

    @pytest.fixture
    def middleware_with_tool(self) -> TurnUrgencyMiddleware:
        """Middleware fixture with required_completion_tool configured."""
        return TurnUrgencyMiddleware(
            TurnUrgencyConfig(max_turns=5, required_completion_tool="recommend_product")
        )

    def test_middleware_initialization(self, middleware: TurnUrgencyMiddleware) -> None:
        """Middleware should initialize with config."""
        assert middleware is not None
        assert middleware.current_urgency is None

    def test_process_messages_injects_urgency(self, middleware: TurnUrgencyMiddleware) -> None:
        """Middleware should inject urgency guidance into messages."""
        request = make_mock_request(messages=[HumanMessage(content="Hello")])
        handler = make_mock_handler()

        middleware.wrap_model_call(request, handler)

        request.override.assert_called_once()
        call_kwargs = request.override.call_args.kwargs
        injected_messages = call_kwargs["messages"]
        steering_msg = injected_messages[-1]
        assert isinstance(steering_msg, SystemMessage)
        assert isinstance(steering_msg.content, str)
        assert "<system>" in steering_msg.content
        assert "URGENCY:" in steering_msg.content
        assert middleware.current_urgency is not None
        assert middleware.current_urgency.current_level == UrgencyLevel.EXPLORATION

    def test_process_messages_appends_steering_message(
        self, middleware: TurnUrgencyMiddleware
    ) -> None:
        """Middleware should append a <system>-tagged message to the list."""
        request = make_mock_request(messages=[HumanMessage(content="Hello")])
        handler = make_mock_handler()

        middleware.wrap_model_call(request, handler)

        request.override.assert_called_once()
        call_kwargs = request.override.call_args.kwargs
        injected_messages = call_kwargs["messages"]
        # Original message + steering message
        assert len(injected_messages) == 2
        assert isinstance(injected_messages[-1], SystemMessage)
        assert "<system>" in injected_messages[-1].content

    def test_urgency_transition_tracking(self, middleware: TurnUrgencyMiddleware) -> None:
        """Middleware should track urgency transitions across calls."""
        # Turn 1 — EXPLORATION (1 human message, 4 remaining of 5 = 80%)
        request1 = make_mock_request(messages=[HumanMessage(content="Hello")])
        middleware.wrap_model_call(request1, make_mock_handler())
        assert middleware.current_urgency is not None
        assert middleware.current_urgency.current_level == UrgencyLevel.EXPLORATION

        # Turn 2 with enough messages to reach BALANCED
        messages2 = [
            HumanMessage(content="1"),
            AIMessage(content="1"),
            HumanMessage(content="2"),
            AIMessage(content="2"),
            HumanMessage(content="3"),
        ]
        request2 = make_mock_request(messages=messages2)
        middleware.wrap_model_call(request2, make_mock_handler())
        assert middleware.current_urgency.current_level == UrgencyLevel.BALANCED

    def test_reset_clears_urgency(self, middleware: TurnUrgencyMiddleware) -> None:
        """Reset should clear urgency tracking."""
        request = make_mock_request(messages=[HumanMessage(content="Hello")])
        middleware.wrap_model_call(request, make_mock_handler())
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

    def test_completion_tool_tracked(
        self, middleware_with_tool: TurnUrgencyMiddleware
    ) -> None:
        """Middleware should track completion_tool_called in Command update when tool fires."""
        recommend_ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "recommend_product",
                    "id": "call_1",
                    "args": {"product_name": "X", "reason": "Y"},
                }
            ],
        )
        handler = make_mock_handler(response_messages=[recommend_ai_msg])
        request = make_mock_request(messages=[HumanMessage(content="What do you recommend?")])

        result = middleware_with_tool.wrap_model_call(request, handler)

        from langchain.agents.middleware import ExtendedModelResponse

        assert isinstance(result, ExtendedModelResponse)
        state_update = result.command.update  # type: ignore[union-attr]
        assert state_update["completion_tool_called"] is True

    def test_terminal_without_completion_forces_guidance(
        self, middleware_with_tool: TurnUrgencyMiddleware
    ) -> None:
        """On terminal turn without completion tool, guidance should contain forceful escalation."""
        # Build 5 turns of messages to trigger TERMINAL level
        terminal_messages = []
        for i in range(5):
            terminal_messages.extend([
                HumanMessage(content=f"question {i}"),
                AIMessage(content=f"answer {i}"),
            ])

        request = make_mock_request(messages=terminal_messages)
        handler = make_mock_handler()

        middleware_with_tool.wrap_model_call(request, handler)

        request.override.assert_called_once()
        call_kwargs = request.override.call_args.kwargs
        injected_messages = call_kwargs["messages"]
        steering_content: str = injected_messages[-1].content
        assert "URGENCY: TERMINAL" in steering_content
        # Forceful escalation line should be present since tool was never called
        assert "MUST" in steering_content or "NOW" in steering_content


class TestIntegration:
    """Integration tests for the complete middleware flow."""

    def test_full_shopping_conversation_flow(self) -> None:
        """Test a complete shopping conversation with urgency escalation."""
        config = TurnUrgencyConfig(
            max_turns=3,
            required_completion_tool="recommend_product",
        )
        middleware = TurnUrgencyMiddleware(config)

        # Turn 1: EXPLORATION (1 human, 2 remaining of 3 = 67%)
        messages1 = [HumanMessage(content="I'm looking for wireless headphones")]
        request1 = make_mock_request(messages=messages1)
        middleware.wrap_model_call(request1, make_mock_handler())
        urgency1 = middleware.current_urgency
        assert urgency1 is not None
        assert urgency1.current_level == UrgencyLevel.EXPLORATION

        # Turn 2: BALANCED (2 turns used, 1 remaining = 33%)
        messages2 = [
            HumanMessage(content="I'm looking for wireless headphones"),
            AIMessage(content="What's your budget?"),
            HumanMessage(content="Around $150"),
        ]
        request2 = make_mock_request(messages=messages2)
        middleware.wrap_model_call(request2, make_mock_handler())
        urgency2 = middleware.current_urgency
        assert urgency2 is not None
        assert urgency2.turns_used == 2
        assert urgency2.current_level == UrgencyLevel.BALANCED

        # Turn 3: TERMINAL (3 turns used, 0 remaining = 0%)
        messages3 = messages2 + [
            AIMessage(content="I'd suggest the AudioMax 500."),
            HumanMessage(content="Sounds good, go ahead."),
        ]
        request3 = make_mock_request(messages=messages3)
        middleware.wrap_model_call(request3, make_mock_handler())
        urgency3 = middleware.current_urgency
        assert urgency3 is not None
        assert urgency3.current_level == UrgencyLevel.TERMINAL
        assert urgency3.is_terminal is True

    def test_custom_thresholds_flow(self) -> None:
        """Test with custom urgency thresholds via wrap_model_call."""
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

        request = make_mock_request(messages=messages)
        middleware.wrap_model_call(request, make_mock_handler())
        urgency = middleware.current_urgency
        assert urgency is not None
        assert urgency.turns_used == 6
        assert urgency.remaining_ratio == 0.4
        assert urgency.current_level == UrgencyLevel.BALANCED

        # Add more messages to reach EFFICIENCY
        messages.extend([
            HumanMessage(content="x"),
            AIMessage(content="x"),
            HumanMessage(content="y"),
            AIMessage(content="y"),
        ])
        request2 = make_mock_request(messages=messages)
        middleware.wrap_model_call(request2, make_mock_handler())
        urgency2 = middleware.current_urgency
        assert urgency2 is not None
        assert urgency2.turns_used == 8
        assert urgency2.remaining_ratio == 0.2
        # At exactly 20%, it's EFFICIENCY (not > 20%)
        assert urgency2.current_level == UrgencyLevel.EFFICIENCY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
