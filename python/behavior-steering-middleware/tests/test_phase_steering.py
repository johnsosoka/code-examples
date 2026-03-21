"""Unit tests for Phase-Based Steering middleware.

This module provides comprehensive tests for all components of the
phase-based steering pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from bsm_examples.phase_steering.config import PhaseConfig
    from bsm_examples.phase_steering.state import PhaseState


class TestPhase:
    """Tests for the Phase enum."""

    def test_phase_values(self) -> None:
        """Test that all expected phases exist."""
        from bsm_examples.phase_steering.phase import Phase

        assert Phase.GATHER is not None
        assert Phase.ANALYZE is not None
        assert Phase.RECOMMEND is not None
        assert Phase.COMPLETE is not None

    def test_phase_order(self) -> None:
        """Test phase ordering is correct."""
        from bsm_examples.phase_steering.phase import Phase

        assert Phase.GATHER.order == 0
        assert Phase.ANALYZE.order == 1
        assert Phase.RECOMMEND.order == 2
        assert Phase.COMPLETE.order == 3

    def test_phase_display_names(self) -> None:
        """Test display names are properly formatted."""
        from bsm_examples.phase_steering.phase import Phase

        assert Phase.GATHER.display_name == "Gather"
        assert Phase.ANALYZE.display_name == "Analyze"
        assert Phase.RECOMMEND.display_name == "Recommend"
        assert Phase.COMPLETE.display_name == "Complete"

    def test_phase_from_string_valid(self) -> None:
        """Test parsing phase from valid strings."""
        from bsm_examples.phase_steering.phase import Phase

        assert Phase.from_string("GATHER") == Phase.GATHER
        assert Phase.from_string("gather") == Phase.GATHER
        assert Phase.from_string("Analyze") == Phase.ANALYZE

    def test_phase_from_string_invalid(self) -> None:
        """Test parsing phase from invalid string raises error."""
        from bsm_examples.phase_steering.phase import Phase

        with pytest.raises(ValueError, match="Unknown phase"):
            Phase.from_string("INVALID")


class TestPhaseBehavior:
    """Tests for PhaseBehavior model."""

    def test_basic_creation(self) -> None:
        """Test creating PhaseBehavior with required fields."""
        from bsm_examples.phase_steering.config import PhaseBehavior

        behavior = PhaseBehavior(system_prompt_addition="Test guidance")
        assert behavior.system_prompt_addition == "Test guidance"
        assert behavior.allowed_tools == []
        assert behavior.max_iterations == 10

    def test_full_creation(self) -> None:
        """Test creating PhaseBehavior with all fields."""
        from bsm_examples.phase_steering.config import PhaseBehavior

        behavior = PhaseBehavior(
            system_prompt_addition="Focus on gathering data",
            allowed_tools=["search", "fetch"],
            completion_criteria="Collect 3 items",
            max_iterations=5,
        )
        assert behavior.system_prompt_addition == "Focus on gathering data"
        assert behavior.allowed_tools == ["search", "fetch"]
        assert behavior.completion_criteria == "Collect 3 items"
        assert behavior.max_iterations == 5

    def test_max_iterations_validation(self) -> None:
        """Test max_iterations validation constraints."""
        from bsm_examples.phase_steering.config import PhaseBehavior

        # Valid values
        PhaseBehavior(system_prompt_addition="Test", max_iterations=1)
        PhaseBehavior(system_prompt_addition="Test", max_iterations=50)

        # Invalid values
        with pytest.raises(ValueError):
            PhaseBehavior(system_prompt_addition="Test", max_iterations=0)
        with pytest.raises(ValueError):
            PhaseBehavior(system_prompt_addition="Test", max_iterations=51)


class TestPhaseTransition:
    """Tests for PhaseTransition model."""

    def test_creation(self) -> None:
        """Test creating PhaseTransition."""
        from bsm_examples.phase_steering.config import PhaseTransition
        from bsm_examples.phase_steering.phase import Phase

        transition = PhaseTransition(
            from_phase=Phase.GATHER,
            to_phase=Phase.ANALYZE,
            condition_description="Data collected",
        )
        assert transition.from_phase == Phase.GATHER
        assert transition.to_phase == Phase.ANALYZE
        assert transition.condition_description == "Data collected"
        assert transition.requires_tool_call is None
        assert transition.requires_state_key is None


class TestPhaseConfig:
    """Tests for PhaseConfig model."""

    @pytest.fixture
    def basic_config(self) -> PhaseConfig:
        """Create a basic phase configuration for testing."""
        from bsm_examples.phase_steering.config import PhaseBehavior, PhaseConfig
        from bsm_examples.phase_steering.phase import Phase

        return PhaseConfig(
            phases={
                Phase.GATHER: PhaseBehavior(
                    system_prompt_addition="Gather data",
                    max_iterations=5,
                ),
                Phase.ANALYZE: PhaseBehavior(
                    system_prompt_addition="Analyze data",
                    max_iterations=3,
                ),
            },
        )

    def test_default_initial_phase(self, basic_config: PhaseConfig) -> None:
        """Test that default initial phase is GATHER."""
        from bsm_examples.phase_steering.phase import Phase

        assert basic_config.initial_phase == Phase.GATHER

    def test_empty_phases_validation(self) -> None:
        """Test that empty phases dict raises validation error."""
        from bsm_examples.phase_steering.config import PhaseConfig

        with pytest.raises(ValueError, match="At least one phase"):
            PhaseConfig(phases={})

    def test_get_allowed_transitions(self, basic_config: PhaseConfig) -> None:
        """Test getting allowed transitions."""
        from bsm_examples.phase_steering.config import PhaseTransition
        from bsm_examples.phase_steering.phase import Phase

        # Add transitions
        basic_config.transitions = [
            PhaseTransition(
                from_phase=Phase.GATHER,
                to_phase=Phase.ANALYZE,
                condition_description="Done gathering",
            ),
        ]

        allowed = basic_config.get_allowed_transitions(Phase.GATHER)
        assert Phase.ANALYZE in allowed
        assert Phase.GATHER not in allowed

    def test_is_valid_transition(self, basic_config: PhaseConfig) -> None:
        """Test checking if transition is valid."""
        from bsm_examples.phase_steering.config import PhaseTransition
        from bsm_examples.phase_steering.phase import Phase

        basic_config.transitions = [
            PhaseTransition(
                from_phase=Phase.GATHER,
                to_phase=Phase.ANALYZE,
                condition_description="Done gathering",
            ),
        ]

        assert basic_config.is_valid_transition(Phase.GATHER, Phase.ANALYZE) is True
        assert basic_config.is_valid_transition(Phase.ANALYZE, Phase.GATHER) is False


class TestPhaseState:
    """Tests for PhaseState model."""

    @pytest.fixture
    def empty_state(self) -> PhaseState:
        """Create an empty PhaseState for testing."""
        from bsm_examples.phase_steering.state import PhaseState

        return PhaseState()

    def test_default_values(self, empty_state: PhaseState) -> None:
        """Test default values of empty state."""
        assert empty_state.current_phase is None
        assert empty_state.phase_iterations == {}
        assert empty_state.phase_history == []
        assert empty_state.gathered_data == []
        assert empty_state.analysis_notes == ""
        assert empty_state.recommendations == []
        assert empty_state.query == ""

    def test_record_phase_entry(self, empty_state: PhaseState) -> None:
        """Test recording phase entry."""
        from bsm_examples.phase_steering.phase import Phase

        empty_state.record_phase_entry(Phase.GATHER)
        assert empty_state.current_phase == Phase.GATHER
        assert empty_state.phase_iterations[Phase.GATHER] == 1

        # Record another entry
        empty_state.record_phase_entry(Phase.ANALYZE)
        assert empty_state.current_phase == Phase.ANALYZE
        assert Phase.GATHER in empty_state.phase_history
        assert empty_state.phase_iterations[Phase.ANALYZE] == 1

    def test_increment_iteration(self, empty_state: PhaseState) -> None:
        """Test incrementing iteration count."""
        from bsm_examples.phase_steering.phase import Phase

        empty_state.current_phase = Phase.GATHER
        empty_state.increment_iteration()
        assert empty_state.phase_iterations[Phase.GATHER] == 1

        empty_state.increment_iteration(Phase.GATHER)
        assert empty_state.phase_iterations[Phase.GATHER] == 2

    def test_add_gathered_data(self, empty_state: PhaseState) -> None:
        """Test adding gathered data."""
        data = {"source": "web", "content": "test"}
        empty_state.add_gathered_data(data)
        assert len(empty_state.gathered_data) == 1
        assert empty_state.gathered_data[0] == data

    def test_set_analysis(self, empty_state: PhaseState) -> None:
        """Test setting analysis notes."""
        notes = "Key findings: ..."
        empty_state.set_analysis(notes)
        assert empty_state.analysis_notes == notes

    def test_add_recommendation(self, empty_state: PhaseState) -> None:
        """Test adding recommendations."""
        rec = "Use OAuth 2.0"
        empty_state.add_recommendation(rec)
        assert len(empty_state.recommendations) == 1
        assert empty_state.recommendations[0] == rec

    def test_is_phase_complete(self, empty_state: PhaseState) -> None:
        """Test phase completion check."""
        from bsm_examples.phase_steering.phase import Phase

        empty_state.phase_iterations[Phase.GATHER] = 3
        assert empty_state.is_phase_complete(Phase.GATHER, min_iterations=2) is True
        assert empty_state.is_phase_complete(Phase.GATHER, min_iterations=5) is False

    def test_total_iterations(self, empty_state: PhaseState) -> None:
        """Test total iterations calculation."""
        from bsm_examples.phase_steering.phase import Phase

        empty_state.phase_iterations = {
            Phase.GATHER: 3,
            Phase.ANALYZE: 2,
        }
        assert empty_state.total_iterations == 5


class TestPhaseDetector:
    """Tests for PhaseDetector implementations."""

    @pytest.fixture
    def config(self) -> PhaseConfig:
        """Create a basic configuration for detector tests."""
        from bsm_examples.phase_steering.config import PhaseBehavior, PhaseConfig
        from bsm_examples.phase_steering.phase import Phase

        return PhaseConfig(
            phases={
                Phase.GATHER: PhaseBehavior(
                    system_prompt_addition="Gather",
                    max_iterations=5,
                ),
                Phase.ANALYZE: PhaseBehavior(
                    system_prompt_addition="Analyze",
                    max_iterations=3,
                ),
                Phase.RECOMMEND: PhaseBehavior(
                    system_prompt_addition="Recommend",
                    max_iterations=2,
                ),
            },
        )

    @pytest.fixture
    def detector(self, config: PhaseConfig):
        """Create a DefaultPhaseDetector instance."""
        from bsm_examples.phase_steering.detector import DefaultPhaseDetector

        return DefaultPhaseDetector(config)

    def test_detect_phase_from_state(self, detector, config: PhaseConfig) -> None:
        """Test detecting phase from explicit state."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        state = PhaseState(current_phase=Phase.ANALYZE)
        detected = detector.detect_phase(state, [])
        assert detected == Phase.ANALYZE

    def test_detect_phase_from_tool_calls(self, detector, config: PhaseConfig) -> None:
        """Test detecting phase from tool call patterns."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        state = PhaseState()
        tool_calls = [{"name": "search"}, {"name": "fetch"}]
        detected = detector.detect_phase(state, [], tool_calls)
        assert detected == Phase.GATHER

        tool_calls = [{"name": "compare"}, {"name": "evaluate"}]
        detected = detector.detect_phase(state, [], tool_calls)
        assert detected == Phase.ANALYZE

        tool_calls = [{"name": "synthesize"}]
        detected = detector.detect_phase(state, [], tool_calls)
        assert detected == Phase.RECOMMEND

    def test_detect_phase_from_state_content(self, detector, config: PhaseConfig) -> None:
        """Test detecting phase from state content."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        # Recommendations present
        state = PhaseState(recommendations=["Use OAuth"])
        detected = detector.detect_phase(state, [])
        assert detected == Phase.RECOMMEND

        # Analysis notes present
        state = PhaseState(analysis_notes="Some analysis here")
        detected = detector.detect_phase(state, [])
        assert detected == Phase.ANALYZE

        # Gathered data present
        state = PhaseState(gathered_data=[{"test": "data"}])
        detected = detector.detect_phase(state, [])
        assert detected == Phase.GATHER

    def test_should_transition_iteration_limit(self, detector, config: PhaseConfig) -> None:
        """Test transition detection based on iteration limit."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        state = PhaseState(phase_iterations={Phase.GATHER: 5})
        should_transition, target = detector.should_transition(Phase.GATHER, state, [])

        assert should_transition is True
        assert target == Phase.ANALYZE

    def test_should_transition_signal_in_message(self, detector, config: PhaseConfig) -> None:
        """Test transition detection from message content."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        state = PhaseState()
        messages = [HumanMessage(content="I'm ready to analyze the data now")]

        should_transition, target = detector.should_transition(Phase.GATHER, state, messages)
        assert should_transition is True
        assert target == Phase.ANALYZE

    def test_should_not_transition(self, detector, config: PhaseConfig) -> None:
        """Test when transition should not occur."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        state = PhaseState(phase_iterations={Phase.GATHER: 1})
        should_transition, target = detector.should_transition(Phase.GATHER, state, [])

        assert should_transition is False
        assert target is None


class TestPhaseSteeringMiddleware:
    """Tests for PhaseSteeringMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a PhaseSteeringMiddleware instance."""
        from bsm_examples.phase_steering.config import PhaseBehavior, PhaseConfig
        from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware
        from bsm_examples.phase_steering.phase import Phase

        config = PhaseConfig(
            phases={
                Phase.GATHER: PhaseBehavior(
                    system_prompt_addition="Focus on gathering data",
                    allowed_tools=["search"],
                    max_iterations=5,
                ),
                Phase.ANALYZE: PhaseBehavior(
                    system_prompt_addition="Analyze the data",
                    allowed_tools=["compare"],
                    max_iterations=3,
                ),
            },
        )
        return PhaseSteeringMiddleware(config)

    def test_initial_state(self, middleware) -> None:
        """Test middleware initial state."""
        from bsm_examples.phase_steering.phase import Phase

        assert middleware.current_phase == Phase.GATHER
        assert middleware.phase_history == []

    def test_process_request_injects_guidance(self, middleware) -> None:
        """Test that process_request injects phase guidance."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        state = PhaseState()
        state.record_phase_entry(Phase.GATHER)

        messages = [HumanMessage(content="Test message")]
        phase, modified_messages = middleware.process_request(state, messages)

        # Check that a system message was added
        system_msgs = [m for m in modified_messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) > 0

        # Check that phase guidance is present
        system_content = system_msgs[0].content
        assert isinstance(system_content, str)
        assert "[PHASE: Gather]" in system_content
        assert "Focus on gathering data" in system_content

    def test_process_request_appends_to_existing_system(self, middleware) -> None:
        """Test that guidance is appended to existing system message."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        state = PhaseState()
        state.record_phase_entry(Phase.GATHER)

        messages = [
            SystemMessage(content="Original system prompt"),
            HumanMessage(content="Test message"),
        ]
        phase, modified_messages = middleware.process_request(state, messages)

        # Should still have only one system message
        system_msgs = [m for m in modified_messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1

        # Content should include both original and phase guidance
        content = system_msgs[0].content
        assert isinstance(content, str)
        assert "Original system prompt" in content
        assert "[PHASE: Gather]" in content

    def test_reset(self, middleware) -> None:
        """Test middleware reset functionality."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        # Process some requests to build state
        state = PhaseState()
        state.record_phase_entry(Phase.GATHER)
        middleware.process_request(state, [HumanMessage(content="Test")])

        # Reset
        middleware.reset()

        # State should be cleared
        assert middleware.current_phase == Phase.GATHER
        assert middleware.phase_history == []

    def test_should_continue(self, middleware) -> None:
        """Test should_continue method."""
        from bsm_examples.phase_steering.phase import Phase
        from bsm_examples.phase_steering.state import PhaseState

        # Not complete
        state = PhaseState(current_phase=Phase.GATHER)
        assert middleware.should_continue(state) is True

        # Complete
        state = PhaseState(current_phase=Phase.COMPLETE)
        assert middleware.should_continue(state) is False


class TestDemo:
    """Tests for demo functionality."""

    def test_create_research_config(self) -> None:
        """Test that demo config is created correctly."""
        from bsm_examples.phase_steering.demo import create_research_config
        from bsm_examples.phase_steering.phase import Phase

        config = create_research_config()

        assert Phase.GATHER in config.phases
        assert Phase.ANALYZE in config.phases
        assert Phase.RECOMMEND in config.phases
        assert Phase.COMPLETE in config.phases

        assert len(config.transitions) > 0
        assert config.initial_phase == Phase.GATHER

    def test_mock_tools(self) -> None:
        """Test mock tool functions."""
        from bsm_examples.phase_steering.demo import mock_compare, mock_search, mock_synthesize

        # Test search
        result = mock_search("test query")
        assert result["tool"] == "search"
        assert "results" in result

        # Test compare
        result = mock_compare(["a", "b"], ["x", "y"])
        assert result["tool"] == "compare"
        assert "comparison" in result

        # Test synthesize
        result = mock_synthesize("analysis", "recommendations")
        assert result["tool"] == "synthesize"
        assert "recommendations" in result
