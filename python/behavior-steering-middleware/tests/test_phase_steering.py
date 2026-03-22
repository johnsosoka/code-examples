"""Unit tests for Phase-Based Steering middleware.

This module provides comprehensive tests for all components of the
phase-based steering pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from tests.helpers import make_mock_handler, make_mock_request

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
        """Test that wrap_model_call appends <system>-tagged phase guidance."""
        request = make_mock_request(messages=[HumanMessage(content="Test")])
        handler = make_mock_handler()

        middleware.wrap_model_call(request, handler)

        request.override.assert_called_once()
        call_kwargs = request.override.call_args.kwargs
        injected_messages = call_kwargs["messages"]
        steering_msg = injected_messages[-1]
        assert isinstance(steering_msg, SystemMessage)
        assert isinstance(steering_msg.content, str)
        assert "<system>" in steering_msg.content
        assert "[PHASE: Gather]" in steering_msg.content
        assert "Focus on gathering data" in steering_msg.content

    def test_process_request_preserves_existing_messages(
        self, middleware
    ) -> None:
        """Test that original messages are preserved with guidance appended."""
        original_msg = HumanMessage(content="Test message")
        request = make_mock_request(messages=[original_msg])
        handler = make_mock_handler()

        middleware.wrap_model_call(request, handler)

        request.override.assert_called_once()
        call_kwargs = request.override.call_args.kwargs
        injected_messages = call_kwargs["messages"]
        assert len(injected_messages) == 2
        assert injected_messages[0] is original_msg
        assert isinstance(injected_messages[-1], SystemMessage)
        assert "<system>" in injected_messages[-1].content

    def test_reset(self, middleware) -> None:
        """Test middleware reset functionality."""
        from bsm_examples.phase_steering.phase import Phase

        # Make a call to advance internal iteration counters
        request = make_mock_request(messages=[HumanMessage(content="Test")])
        middleware.wrap_model_call(request, make_mock_handler())

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
        """Test demo tool functions return expected content."""
        from bsm_examples.phase_steering.demo import compare, search, synthesize

        # Test search returns a string mentioning the query
        result = search.invoke({"query": "test query"})
        assert isinstance(result, str)
        assert "test query" in result

        # Test compare returns a string mentioning the items
        result = compare.invoke({"items": "a, b"})
        assert isinstance(result, str)
        assert "Comparison" in result

        # Test synthesize returns a string with recommendations
        result = synthesize.invoke({"analysis": "some analysis"})
        assert isinstance(result, str)
        assert "recommendation" in result.lower()
