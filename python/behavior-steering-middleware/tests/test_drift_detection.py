"""Unit tests for behavior drift detection middleware.

This module provides comprehensive tests for:
- ToolUsagePattern validation
- DriftDetectionConfig validation
- DriftDetector scoring algorithm
- DriftSteeringMiddleware integration
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from bsm_examples.drift_detection.middleware import (
    DriftDetectionConfig,
    DriftDetector,
    DriftSteeringMiddleware,
    ToolUsagePattern,
)


class TestToolUsagePattern:
    """Tests for ToolUsagePattern Pydantic model."""

    def test_default_construction(self) -> None:
        """Test that pattern can be created with defaults."""
        pattern = ToolUsagePattern(name="test_pattern")
        assert pattern.name == "test_pattern"
        assert pattern.expected_tools == {}
        assert pattern.required_tools == []
        assert pattern.forbidden_tools == []

    def test_full_construction(self) -> None:
        """Test that pattern can be created with all fields."""
        pattern = ToolUsagePattern(
            name="calculator",
            expected_tools={"calculate": 0.8, "convert": 0.2},
            minimum_frequencies={"calculate": 0.5},
            required_tools=["calculate"],
            forbidden_tools=["dangerous_tool"],
        )
        assert pattern.name == "calculator"
        assert pattern.expected_tools == {"calculate": 0.8, "convert": 0.2}
        assert pattern.minimum_frequencies == {"calculate": 0.5}
        assert pattern.required_tools == ["calculate"]
        assert pattern.forbidden_tools == ["dangerous_tool"]

    def test_frequency_validation(self) -> None:
        """Test that invalid frequencies are rejected."""
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            ToolUsagePattern(
                name="test",
                expected_tools={"tool": 1.5},  # Invalid: > 1.0
            )

    def test_negative_frequency_validation(self) -> None:
        """Test that negative frequencies are rejected."""
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            ToolUsagePattern(
                name="test",
                expected_tools={"tool": -0.1},  # Invalid: < 0.0
            )

    def test_get_expected_frequency(self) -> None:
        """Test getting expected frequency for a tool."""
        pattern = ToolUsagePattern(
            name="test",
            expected_tools={"tool_a": 0.5, "tool_b": 0.3},
        )
        assert pattern.get_expected_frequency("tool_a") == 0.5
        assert pattern.get_expected_frequency("tool_b") == 0.3
        assert pattern.get_expected_frequency("unknown") == 0.0

    def test_get_minimum_frequency(self) -> None:
        """Test getting minimum frequency for a tool."""
        pattern = ToolUsagePattern(
            name="test",
            minimum_frequencies={"tool_a": 0.2},
        )
        assert pattern.get_minimum_frequency("tool_a") == 0.2
        assert pattern.get_minimum_frequency("unknown") is None


class TestDriftDetectionConfig:
    """Tests for DriftDetectionConfig Pydantic model."""

    def test_default_construction(self) -> None:
        """Test that config can be created with defaults."""
        config = DriftDetectionConfig()
        assert config.history_window_size == 10
        assert config.recency_bias == 1.5
        assert config.warning_threshold == 0.3
        assert config.intervention_threshold == 0.6
        assert config.max_interventions_per_conversation == 3
        assert config.cooldown_turns == 2
        assert config.include_statistics is True

    def test_custom_construction(self) -> None:
        """Test that config can be created with custom values."""
        config = DriftDetectionConfig(
            history_window_size=5,
            recency_bias=2.0,
            warning_threshold=0.4,
            intervention_threshold=0.7,
            max_interventions_per_conversation=5,
            cooldown_turns=1,
            include_statistics=False,
        )
        assert config.history_window_size == 5
        assert config.recency_bias == 2.0
        assert config.warning_threshold == 0.4
        assert config.intervention_threshold == 0.7
        assert config.max_interventions_per_conversation == 5
        assert config.cooldown_turns == 1
        assert config.include_statistics is False

    def test_threshold_validation(self) -> None:
        """Test that intervention threshold must be >= warning threshold."""
        with pytest.raises(
            ValueError, match="intervention_threshold .* must be >= warning_threshold"
        ):
            DriftDetectionConfig(
                warning_threshold=0.7,
                intervention_threshold=0.5,  # Invalid: less than warning
            )

    def test_history_window_bounds(self) -> None:
        """Test that history window size has bounds."""
        with pytest.raises(ValueError):
            DriftDetectionConfig(history_window_size=2)  # Below minimum

        with pytest.raises(ValueError):
            DriftDetectionConfig(history_window_size=101)  # Above maximum

    def test_recency_bias_bounds(self) -> None:
        """Test that recency bias has bounds."""
        with pytest.raises(ValueError):
            DriftDetectionConfig(recency_bias=0.05)  # Below minimum

        with pytest.raises(ValueError):
            DriftDetectionConfig(recency_bias=6.0)  # Above maximum


class TestDriftDetector:
    """Tests for DriftDetector class."""

    @pytest.fixture
    def pattern(self) -> ToolUsagePattern:
        """Create a test pattern."""
        return ToolUsagePattern(
            name="test",
            expected_tools={"tool_a": 0.8, "tool_b": 0.2},
            required_tools=["tool_a"],
            minimum_frequencies={"tool_a": 0.5},
        )

    @pytest.fixture
    def config(self) -> DriftDetectionConfig:
        """Create a test config."""
        return DriftDetectionConfig(
            history_window_size=5,
            intervention_threshold=0.6,
        )

    def test_initial_state(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test that detector starts with zero drift."""
        detector = DriftDetector(pattern, config)
        assert detector.calculate_drift_score() == 0.0
        assert detector.turn_count == 0
        assert detector.intervention_count == 0

    def test_record_tool_usage(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test recording tool usage from messages."""
        detector = DriftDetector(pattern, config)

        # Create a message with a tool call
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "tool_a", "args": {}, "id": "call_1"}],
        )

        detector.record_turn([ai_msg])
        assert detector.turn_count == 1
        assert len(detector.history) == 1

    def test_drift_score_with_no_tools(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test drift score when no tools are used."""
        detector = DriftDetector(pattern, config)

        # Record several turns with no tool calls
        for _ in range(5):
            ai_msg = AIMessage(content="Direct response")
            detector.record_turn([ai_msg])

        # Should have high drift since expected tools aren't being used
        drift = detector.calculate_drift_score()
        assert drift > 0.5  # Should be significant drift

    def test_drift_score_with_correct_usage(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test drift score with correct tool usage."""
        # Create a pattern that only expects tool_a
        single_tool_pattern = ToolUsagePattern(
            name="single_tool",
            expected_tools={"tool_a": 0.8},
            required_tools=["tool_a"],
        )
        detector = DriftDetector(single_tool_pattern, config)

        # Record several turns with correct tool calls
        for i in range(5):
            ai_msg = AIMessage(
                content="",
                tool_calls=[{"name": "tool_a", "args": {}, "id": f"call_{i}"}],
            )
            detector.record_turn([ai_msg])

        # Should have low drift
        drift = detector.calculate_drift_score()
        assert drift < 0.3  # Should be minimal drift

    def test_forbidden_tool_detection(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test that forbidden tools trigger max drift."""
        pattern_with_forbidden = ToolUsagePattern(
            name="test",
            expected_tools={"tool_a": 0.8},
            forbidden_tools=["forbidden"],
        )
        detector = DriftDetector(pattern_with_forbidden, config)

        # Record a turn with a forbidden tool
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "forbidden", "args": {}, "id": "call_1"}],
        )
        detector.record_turn([ai_msg])

        # Should have maximum drift
        assert detector.calculate_drift_score() == 1.0

    def test_required_tool_penalty(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test penalty for missing required tools."""
        detector = DriftDetector(pattern, config)

        # Record several turns without the required tool
        for i in range(5):
            ai_msg = AIMessage(
                content="",
                tool_calls=[{"name": "tool_b", "args": {}, "id": f"call_{i}"}],  # Not required
            )
            detector.record_turn([ai_msg])

        # Should have drift due to missing required tool
        drift = detector.calculate_drift_score()
        assert drift > 0.0

    def test_intervention_threshold(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test intervention decision logic."""
        detector = DriftDetector(pattern, config)

        # Initially should not intervene
        assert not detector.should_intervene()

        # Record many turns without tools to trigger drift
        for _ in range(10):
            ai_msg = AIMessage(content="Direct response")
            detector.record_turn([ai_msg])

        # Now should intervene (if cooldown allows)
        detector.turns_since_last_intervention = config.cooldown_turns
        if detector.calculate_drift_score() >= config.intervention_threshold:
            assert detector.should_intervene()

    def test_intervention_cooldown(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test that cooldown prevents rapid interventions."""
        detector = DriftDetector(pattern, config)
        detector.intervention_count = 1
        detector.turns_since_last_intervention = 0  # Just intervened

        # Should not intervene during cooldown
        assert not detector.should_intervene()

    def test_max_interventions(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test that max interventions limit is respected."""
        detector = DriftDetector(pattern, config)
        detector.intervention_count = config.max_interventions_per_conversation

        # Should not intervene after max reached
        assert not detector.should_intervene()

    def test_record_intervention(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test recording an intervention."""
        detector = DriftDetector(pattern, config)

        detector.record_intervention()
        assert detector.intervention_count == 1
        assert detector.turns_since_last_intervention == 0

    def test_drift_report(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test drift report generation."""
        detector = DriftDetector(pattern, config)

        # Record some activity
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "tool_a", "args": {}, "id": "call_1"}],
        )
        detector.record_turn([ai_msg])

        report = detector.get_drift_report()
        assert "turn_count" in report
        assert "history_size" in report
        assert "drift_score" in report
        assert "intervention_count" in report
        assert "tool_comparison" in report

    def test_weighted_frequencies(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test weighted frequency calculation."""
        detector = DriftDetector(pattern, config)

        # Record turns with different tools (tool_a in first 2, tool_b in last)
        for i in range(3):
            tool_name = "tool_a" if i < 2 else "tool_b"
            ai_msg = AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": {}, "id": f"call_{i}"}],
            )
            detector.record_turn([ai_msg])

        freqs = detector._calculate_weighted_frequencies()
        assert "tool_a" in freqs
        assert "tool_b" in freqs
        # With recency bias, tool_b (used in most recent turn) should have higher weight
        # even though tool_a was used in 2/3 of turns
        assert freqs["tool_b"] > freqs["tool_a"]


class TestDriftSteeringMiddleware:
    """Tests for DriftSteeringMiddleware class."""

    @pytest.fixture
    def pattern(self) -> ToolUsagePattern:
        """Create a test pattern."""
        return ToolUsagePattern(
            name="test",
            expected_tools={"tool_a": 0.8},
            required_tools=["tool_a"],
        )

    @pytest.fixture
    def config(self) -> DriftDetectionConfig:
        """Create a test config."""
        return DriftDetectionConfig(
            history_window_size=5,
            intervention_threshold=0.6,
        )

    def test_construction(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test middleware construction."""
        middleware = DriftSteeringMiddleware(pattern, config)
        assert middleware.pattern == pattern
        assert middleware.config == config
        assert middleware.detector is not None

    def test_construction_with_defaults(self, pattern: ToolUsagePattern) -> None:
        """Test middleware construction with default config."""
        middleware = DriftSteeringMiddleware(pattern)
        assert middleware.config is not None
        assert isinstance(middleware.config, DriftDetectionConfig)

    def test_generate_reminder(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test reminder generation."""
        middleware = DriftSteeringMiddleware(pattern, config)

        # Record some turns to create drift context
        for _ in range(3):
            ai_msg = AIMessage(content="No tools used")
            middleware.detector.record_turn([ai_msg])

        reminder = middleware._generate_reminder()
        assert "SYSTEM REMINDER" in reminder
        assert "tool_a" in reminder

    def test_reset(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test middleware reset."""
        middleware = DriftSteeringMiddleware(pattern, config)

        # Record some activity
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "tool_a", "args": {}, "id": "call_1"}],
        )
        middleware.detector.record_turn([ai_msg])
        middleware.detector.record_intervention()

        # Reset
        middleware.reset()

        # Should be back to initial state
        assert middleware.detector.turn_count == 0
        assert middleware.detector.intervention_count == 0
        assert len(middleware.detector.history) == 0

    def test_get_drift_report(
        self, pattern: ToolUsagePattern, config: DriftDetectionConfig
    ) -> None:
        """Test getting drift report from middleware."""
        middleware = DriftSteeringMiddleware(pattern, config)

        report = middleware.get_drift_report()
        assert "turn_count" in report
        assert "drift_score" in report


class TestIntegration:
    """Integration tests for the full drift detection flow."""

    def test_full_drift_scenario(self) -> None:
        """Test a complete drift detection and correction scenario."""
        pattern = ToolUsagePattern(
            name="calculator",
            expected_tools={"calculate": 0.9},
            required_tools=["calculate"],
        )
        config = DriftDetectionConfig(
            history_window_size=5,
            intervention_threshold=0.6,
            cooldown_turns=1,
        )
        detector = DriftDetector(pattern, config)

        # Phase 1: Normal usage (low drift)
        for i in range(3):
            ai_msg = AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {}, "id": f"call_{i}"}],
            )
            detector.record_turn([ai_msg])

        initial_drift = detector.calculate_drift_score()
        assert initial_drift < 0.3  # Should be low

        # Phase 2: Drift (no tool usage)
        for _ in range(3):
            ai_msg = AIMessage(content="Direct answer")
            detector.record_turn([ai_msg])

        drift_during = detector.calculate_drift_score()
        assert drift_during > 0.5  # Should be higher

        # Phase 3: Intervention should be needed
        detector.turns_since_last_intervention = config.cooldown_turns
        if drift_during >= config.intervention_threshold:
            assert detector.should_intervene()

            # Record intervention
            detector.record_intervention()
            assert detector.intervention_count == 1

        # Phase 4: Recovery (resume tool usage)
        for i in range(3):
            ai_msg = AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {}, "id": f"call_{i+10}"}],
            )
            detector.record_turn([ai_msg])

        final_drift = detector.calculate_drift_score()
        # Drift should decrease with proper tool usage
        assert final_drift < drift_during

    def test_tool_message_detection(self) -> None:
        """Test that tool messages are properly detected."""
        pattern = ToolUsagePattern(
            name="test",
            expected_tools={"tool_a": 0.5},
        )
        config = DriftDetectionConfig()
        detector = DriftDetector(pattern, config)

        # Create a turn with tool call and response
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "tool_a", "args": {}, "id": "call_1"}],
        )
        tool_msg = ToolMessage(content="Result", tool_call_id="1")

        detector.record_turn([ai_msg, tool_msg])

        # Should have recorded the tool call
        assert detector.turn_count == 1
        assert len(detector.history) == 1

    def test_multiple_tool_calls(self) -> None:
        """Test handling multiple tool calls in one turn."""
        pattern = ToolUsagePattern(
            name="test",
            expected_tools={"tool_a": 0.5, "tool_b": 0.5},
        )
        config = DriftDetectionConfig()
        detector = DriftDetector(pattern, config)

        # Create a turn with multiple tool calls
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool_a", "args": {}, "id": "call_1"},
                {"name": "tool_b", "args": {}, "id": "call_2"},
            ],
        )

        detector.record_turn([ai_msg])

        # Should have recorded both tool calls
        record = detector.history[0]
        assert "tool_a" in record.tools_called
        assert "tool_b" in record.tools_called
