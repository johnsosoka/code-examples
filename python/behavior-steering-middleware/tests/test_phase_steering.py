"""Unit tests for the Tool-Triggered Phase Steering middleware.

Full rewrite covering models, detector, middleware, and integration scenarios.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from tests.helpers import (
    make_mock_handler,
    make_mock_request,
    make_mock_tool_handler,
    make_mock_tool_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*, include_all_phases: bool = True):
    """Build a minimal PhaseConfig for use in tests."""
    from bsm_examples.phase_steering.models import (
        Phase,
        PhaseBehavior,
        PhaseConfig,
        PhaseTransitionRule,
    )

    phases = {
        Phase.DISCOVERY: PhaseBehavior(
            system_prompt_addition="Discover customer needs.",
            allowed_tools=["browse_catalog", "note_preferences"],
            max_iterations=5,
        ),
        Phase.CURATION: PhaseBehavior(
            system_prompt_addition="Curate product options.",
            allowed_tools=["filter_products", "shortlist_items"],
            max_iterations=5,
        ),
    }

    if include_all_phases:
        phases[Phase.RECOMMENDATION] = PhaseBehavior(
            system_prompt_addition="Present recommendation.",
            allowed_tools=["present_recommendation"],
            max_iterations=2,
        )
        phases[Phase.COMPLETE] = PhaseBehavior(
            system_prompt_addition="Workflow complete.",
            allowed_tools=[],
            max_iterations=1,
        )

    transitions = [
        PhaseTransitionRule(
            from_phase=Phase.DISCOVERY,
            to_phase=Phase.CURATION,
            trigger_tool="note_preferences",
        ),
        PhaseTransitionRule(
            from_phase=Phase.CURATION,
            to_phase=Phase.RECOMMENDATION,
            trigger_tool="shortlist_items",
        ),
        PhaseTransitionRule(
            from_phase=Phase.RECOMMENDATION,
            to_phase=Phase.COMPLETE,
            trigger_tool="present_recommendation",
        ),
    ]

    return PhaseConfig(
        phases=phases,
        transitions=transitions,
        initial_phase=Phase.DISCOVERY,
    )


# ---------------------------------------------------------------------------
# TestPhase
# ---------------------------------------------------------------------------


class TestPhase:
    """Tests for the Phase enum."""

    def test_all_phases_exist(self) -> None:
        """DISCOVERY, CURATION, RECOMMENDATION, and COMPLETE must all be defined."""
        from bsm_examples.phase_steering.models import Phase

        assert Phase.DISCOVERY is not None
        assert Phase.CURATION is not None
        assert Phase.RECOMMENDATION is not None
        assert Phase.COMPLETE is not None

    def test_phase_ordering(self) -> None:
        """Phases should have ascending order values."""
        from bsm_examples.phase_steering.models import Phase

        assert Phase.DISCOVERY.order == 0
        assert Phase.CURATION.order == 1
        assert Phase.RECOMMENDATION.order == 2
        assert Phase.COMPLETE.order == 3

    def test_display_name(self) -> None:
        """display_name should be title-cased."""
        from bsm_examples.phase_steering.models import Phase

        assert Phase.DISCOVERY.display_name == "Discovery"
        assert Phase.CURATION.display_name == "Curation"
        assert Phase.RECOMMENDATION.display_name == "Recommendation"
        assert Phase.COMPLETE.display_name == "Complete"

    def test_from_string_valid(self) -> None:
        """from_string should parse valid names case-insensitively."""
        from bsm_examples.phase_steering.models import Phase

        assert Phase.from_string("DISCOVERY") == Phase.DISCOVERY
        assert Phase.from_string("discovery") == Phase.DISCOVERY
        assert Phase.from_string("Curation") == Phase.CURATION
        assert Phase.from_string("COMPLETE") == Phase.COMPLETE

    def test_from_string_invalid(self) -> None:
        """from_string should raise ValueError for unknown names."""
        from bsm_examples.phase_steering.models import Phase

        with pytest.raises(ValueError, match="Unknown phase"):
            Phase.from_string("GATHER")

        with pytest.raises(ValueError, match="Unknown phase"):
            Phase.from_string("INVALID_PHASE")


# ---------------------------------------------------------------------------
# TestPhaseBehavior
# ---------------------------------------------------------------------------


class TestPhaseBehavior:
    """Tests for the PhaseBehavior model."""

    def test_basic_creation(self) -> None:
        """Required field only — defaults should apply."""
        from bsm_examples.phase_steering.models import PhaseBehavior

        behavior = PhaseBehavior(system_prompt_addition="Do something.")
        assert behavior.system_prompt_addition == "Do something."
        assert behavior.allowed_tools == []
        assert behavior.completion_criteria == ""
        assert behavior.max_iterations == 10

    def test_full_creation(self) -> None:
        """All fields should round-trip correctly."""
        from bsm_examples.phase_steering.models import PhaseBehavior

        behavior = PhaseBehavior(
            system_prompt_addition="Browse the catalog.",
            allowed_tools=["browse_catalog", "note_preferences"],
            completion_criteria="Preferences captured",
            max_iterations=5,
        )
        assert behavior.allowed_tools == ["browse_catalog", "note_preferences"]
        assert behavior.completion_criteria == "Preferences captured"
        assert behavior.max_iterations == 5

    def test_max_iterations_lower_bound(self) -> None:
        """max_iterations must be at least 1."""
        from bsm_examples.phase_steering.models import PhaseBehavior

        with pytest.raises(ValueError):
            PhaseBehavior(system_prompt_addition="x", max_iterations=0)

    def test_max_iterations_upper_bound(self) -> None:
        """max_iterations must not exceed 50."""
        from bsm_examples.phase_steering.models import PhaseBehavior

        with pytest.raises(ValueError):
            PhaseBehavior(system_prompt_addition="x", max_iterations=51)

    def test_max_iterations_boundary_values_valid(self) -> None:
        """Boundary values 1 and 50 should be accepted."""
        from bsm_examples.phase_steering.models import PhaseBehavior

        PhaseBehavior(system_prompt_addition="x", max_iterations=1)
        PhaseBehavior(system_prompt_addition="x", max_iterations=50)


# ---------------------------------------------------------------------------
# TestPhaseTransitionRule
# ---------------------------------------------------------------------------


class TestPhaseTransitionRule:
    """Tests for the PhaseTransitionRule model."""

    def test_creation_with_trigger_tool(self) -> None:
        """Rule should store all fields correctly."""
        from bsm_examples.phase_steering.models import Phase, PhaseTransitionRule

        rule = PhaseTransitionRule(
            from_phase=Phase.DISCOVERY,
            to_phase=Phase.CURATION,
            trigger_tool="note_preferences",
            description="Customer needs documented",
        )
        assert rule.from_phase == Phase.DISCOVERY
        assert rule.to_phase == Phase.CURATION
        assert rule.trigger_tool == "note_preferences"
        assert rule.description == "Customer needs documented"

    def test_creation_minimal(self) -> None:
        """description is optional and defaults to empty string."""
        from bsm_examples.phase_steering.models import Phase, PhaseTransitionRule

        rule = PhaseTransitionRule(
            from_phase=Phase.CURATION,
            to_phase=Phase.RECOMMENDATION,
            trigger_tool="shortlist_items",
        )
        assert rule.description == ""


# ---------------------------------------------------------------------------
# TestPhaseConfig
# ---------------------------------------------------------------------------


class TestPhaseConfig:
    """Tests for PhaseConfig validation and lookup methods."""

    def test_empty_phases_validation(self) -> None:
        """PhaseConfig with no phases should raise a validation error."""
        from bsm_examples.phase_steering.models import PhaseConfig

        with pytest.raises(ValueError, match="At least one phase"):
            PhaseConfig(phases={})

    def test_get_transition_for_tool_matching(self) -> None:
        """Should return the rule when phase and tool match."""
        from bsm_examples.phase_steering.models import Phase

        config = _make_config()
        rule = config.get_transition_for_tool(Phase.DISCOVERY, "note_preferences")

        assert rule is not None
        assert rule.to_phase == Phase.CURATION
        assert rule.trigger_tool == "note_preferences"

    def test_get_transition_for_tool_no_match(self) -> None:
        """Should return None when the tool does not match any rule."""
        from bsm_examples.phase_steering.models import Phase

        config = _make_config()
        result = config.get_transition_for_tool(Phase.DISCOVERY, "browse_catalog")

        assert result is None

    def test_get_transition_for_tool_wrong_phase(self) -> None:
        """Should return None when the tool matches but the phase does not."""
        from bsm_examples.phase_steering.models import Phase

        config = _make_config()
        # note_preferences triggers DISCOVERY -> CURATION, but we are in CURATION now
        result = config.get_transition_for_tool(Phase.CURATION, "note_preferences")

        assert result is None


# ---------------------------------------------------------------------------
# TestToolTriggeredDetector
# ---------------------------------------------------------------------------


class TestToolTriggeredDetector:
    """Tests for the ToolTriggeredDetector pure lookup class."""

    def test_check_tool_transition_matching(self) -> None:
        """Returns target phase when a rule matches."""
        from bsm_examples.phase_steering.detector import ToolTriggeredDetector
        from bsm_examples.phase_steering.models import Phase

        config = _make_config()
        detector = ToolTriggeredDetector(config)

        result = detector.check_tool_transition(Phase.DISCOVERY, "note_preferences")

        assert result == Phase.CURATION

    def test_check_tool_transition_no_match(self) -> None:
        """Returns None when no rule matches for the given tool."""
        from bsm_examples.phase_steering.detector import ToolTriggeredDetector
        from bsm_examples.phase_steering.models import Phase

        config = _make_config()
        detector = ToolTriggeredDetector(config)

        result = detector.check_tool_transition(Phase.DISCOVERY, "browse_catalog")

        assert result is None

    def test_check_tool_transition_curation_to_recommendation(self) -> None:
        """Returns RECOMMENDATION when shortlist_items is called in CURATION."""
        from bsm_examples.phase_steering.detector import ToolTriggeredDetector
        from bsm_examples.phase_steering.models import Phase

        config = _make_config()
        detector = ToolTriggeredDetector(config)

        result = detector.check_tool_transition(Phase.CURATION, "shortlist_items")

        assert result == Phase.RECOMMENDATION


# ---------------------------------------------------------------------------
# TestPhaseSteeringMiddleware
# ---------------------------------------------------------------------------


class TestPhaseSteeringMiddleware:
    """Tests for PhaseSteeringMiddleware wrap hooks and state management."""

    @pytest.fixture
    def middleware(self):
        """Fresh middleware instance starting in DISCOVERY."""
        from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware

        return PhaseSteeringMiddleware(_make_config())

    def test_initial_state(self, middleware) -> None:
        """Middleware should start in DISCOVERY with no history."""
        from bsm_examples.phase_steering.models import Phase

        assert middleware.current_phase == Phase.DISCOVERY
        assert middleware.phase_history == []

    def test_wrap_model_call_injects_guidance(self, middleware) -> None:
        """wrap_model_call must inject framework_instruction into HumanMessage."""
        request = make_mock_request(messages=[HumanMessage(content="Hello")])
        handler = make_mock_handler()

        middleware.wrap_model_call(request, handler)

        request.override.assert_called_once()
        injected_messages = request.override.call_args.kwargs["messages"]
        human_msg = injected_messages[0]
        assert isinstance(human_msg, HumanMessage)
        assert "<framework_instruction>" in human_msg.content
        assert "[PHASE: Discovery]" in human_msg.content
        assert "Discover customer needs" in human_msg.content

    def test_wrap_model_call_increments_iterations(self, middleware) -> None:
        """Each wrap_model_call should increment the total iteration counter."""
        request = make_mock_request(messages=[HumanMessage(content="Hello")])

        middleware.wrap_model_call(request, make_mock_handler())
        assert middleware._total_iterations == 1

        middleware.wrap_model_call(request, make_mock_handler())
        assert middleware._total_iterations == 2

    def test_wrap_tool_call_allows_permitted_tool(self, middleware) -> None:
        """browse_catalog is allowed in DISCOVERY — handler should be called."""
        from langchain_core.messages import ToolMessage

        tool_request = make_mock_tool_request("browse_catalog")
        handler = make_mock_tool_handler(content="catalog results", tool_call_id="test_id")

        result = middleware.wrap_tool_call(tool_request, handler)

        handler.assert_called_once_with(tool_request)
        assert isinstance(result, ToolMessage)

    def test_wrap_tool_call_blocks_unpermitted_tool(self, middleware) -> None:
        """filter_products is not in DISCOVERY's allow-list — should return error ToolMessage."""
        from langchain_core.messages import ToolMessage

        tool_request = make_mock_tool_request("filter_products", tool_call_id="blocked_id")
        handler = make_mock_tool_handler()

        result = middleware.wrap_tool_call(tool_request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "not available" in result.content
        assert "filter_products" in result.content
        assert "Discovery" in result.content

    def test_wrap_tool_call_triggers_transition(self, middleware) -> None:
        """Calling note_preferences in DISCOVERY should transition to CURATION."""
        from bsm_examples.phase_steering.models import Phase

        tool_request = make_mock_tool_request("note_preferences", tool_call_id="pref_id")
        handler = make_mock_tool_handler(content="Preferences noted", tool_call_id="pref_id")

        middleware.wrap_tool_call(tool_request, handler)

        assert middleware.current_phase == Phase.CURATION

    def test_wrap_tool_call_records_history_on_transition(self, middleware) -> None:
        """Phase history should record the phase that was left on transition."""
        from bsm_examples.phase_steering.models import Phase

        tool_request = make_mock_tool_request("note_preferences", tool_call_id="pref_id")
        handler = make_mock_tool_handler(content="ok", tool_call_id="pref_id")

        middleware.wrap_tool_call(tool_request, handler)

        assert len(middleware.phase_history) == 1
        phase_left, _ = middleware.phase_history[0]
        assert phase_left == Phase.DISCOVERY

    def test_wrap_tool_call_no_transition_for_non_trigger(self, middleware) -> None:
        """browse_catalog should not trigger any transition in DISCOVERY."""
        from bsm_examples.phase_steering.models import Phase

        tool_request = make_mock_tool_request("browse_catalog", tool_call_id="cat_id")
        handler = make_mock_tool_handler(content="catalog", tool_call_id="cat_id")

        middleware.wrap_tool_call(tool_request, handler)

        assert middleware.current_phase == Phase.DISCOVERY
        assert middleware.phase_history == []

    def test_reset(self, middleware) -> None:
        """reset() should restore middleware to its initial state."""
        from bsm_examples.phase_steering.models import Phase

        # Drive a transition to CURATION
        tool_request = make_mock_tool_request("note_preferences", tool_call_id="x")
        handler = make_mock_tool_handler(content="ok", tool_call_id="x")
        middleware.wrap_tool_call(tool_request, handler)

        assert middleware.current_phase == Phase.CURATION

        middleware.reset()

        assert middleware.current_phase == Phase.DISCOVERY
        assert middleware.phase_history == []
        assert middleware._total_iterations == 0
        assert middleware._phase_iterations == {}

    def test_phase_history_is_a_copy(self, middleware) -> None:
        """phase_history should return a copy, not the internal list."""
        h1 = middleware.phase_history
        h2 = middleware.phase_history
        assert h1 is not h2


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end workflow tests covering the full DISCOVERY -> COMPLETE path."""

    def test_full_workflow_transitions(self) -> None:
        """Full tool-triggered workflow should traverse all four phases in order."""
        from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware
        from bsm_examples.phase_steering.models import Phase

        config = _make_config()
        middleware = PhaseSteeringMiddleware(config)

        # DISCOVERY phase — trigger with note_preferences
        assert middleware.current_phase == Phase.DISCOVERY

        note_req = make_mock_tool_request("note_preferences", tool_call_id="t1")
        note_handler = make_mock_tool_handler(content="ok", tool_call_id="t1")
        middleware.wrap_tool_call(note_req, note_handler)

        assert middleware.current_phase == Phase.CURATION

        # CURATION phase — trigger with shortlist_items
        shortlist_req = make_mock_tool_request("shortlist_items", tool_call_id="t2")
        shortlist_handler = make_mock_tool_handler(content="ok", tool_call_id="t2")
        middleware.wrap_tool_call(shortlist_req, shortlist_handler)

        assert middleware.current_phase == Phase.RECOMMENDATION

        # RECOMMENDATION phase — trigger with present_recommendation
        rec_req = make_mock_tool_request("present_recommendation", tool_call_id="t3")
        rec_handler = make_mock_tool_handler(content="ok", tool_call_id="t3")
        middleware.wrap_tool_call(rec_req, rec_handler)

        assert middleware.current_phase == Phase.COMPLETE

        # History should record all three transitions
        assert len(middleware.phase_history) == 3
        phases_left = [p.name for p, _ in middleware.phase_history]
        assert phases_left == ["DISCOVERY", "CURATION", "RECOMMENDATION"]

    def test_model_call_reflects_current_phase_after_transition(self) -> None:
        """After a tool-triggered transition, wrap_model_call should inject CURATION guidance."""
        from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware

        config = _make_config()
        middleware = PhaseSteeringMiddleware(config)

        # Trigger transition to CURATION
        note_req = make_mock_tool_request("note_preferences", tool_call_id="x")
        note_handler = make_mock_tool_handler(content="ok", tool_call_id="x")
        middleware.wrap_tool_call(note_req, note_handler)

        # Now wrap_model_call should inject CURATION guidance
        request = make_mock_request(messages=[HumanMessage(content="Now curate.")])
        handler = make_mock_handler()

        middleware.wrap_model_call(request, handler)

        injected_messages = request.override.call_args.kwargs["messages"]
        human_msg = injected_messages[0]
        assert isinstance(human_msg, HumanMessage)
        assert "[PHASE: Curation]" in human_msg.content
        assert "Curate product options" in human_msg.content

    def test_blocked_tool_does_not_transition(self) -> None:
        """A blocked (not-allowed) tool call must not trigger a phase transition."""
        from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware
        from bsm_examples.phase_steering.models import Phase

        config = _make_config()
        middleware = PhaseSteeringMiddleware(config)

        # filter_products is NOT in DISCOVERY allow-list
        tool_req = make_mock_tool_request("filter_products", tool_call_id="blocked")
        handler = make_mock_tool_handler(content="should not run", tool_call_id="blocked")

        middleware.wrap_tool_call(tool_req, handler)

        # Phase must still be DISCOVERY
        assert middleware.current_phase == Phase.DISCOVERY
        handler.assert_not_called()
