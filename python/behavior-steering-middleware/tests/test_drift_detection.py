"""Tests for the Worker Check-In Monitor middleware (drift_detection package).

Covers:
- CheckInConfig model validation
- CheckInDetector logic (record_turn, should_intervene, cooldown, etc.)
- CheckInMiddleware wrap_model_call integration
- Full multi-turn scenario
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from bsm_examples.drift_detection.detector import CheckInDetector
from bsm_examples.drift_detection.middleware import CheckInMiddleware
from bsm_examples.drift_detection.models import CheckInConfig
from tests.helpers import make_mock_handler, make_mock_request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ai_with_tool(tool_name: str) -> AIMessage:
    """Return an AIMessage containing a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {}, "id": "tc_1", "type": "tool_call"}],
    )


def _make_ai_no_tool() -> AIMessage:
    """Return an AIMessage with no tool calls."""
    return AIMessage(content="I am still working on the task.")


# ---------------------------------------------------------------------------
# TestCheckInConfig
# ---------------------------------------------------------------------------


class TestCheckInConfig:
    """Pydantic model validation for CheckInConfig."""

    def test_default_values(self) -> None:
        config = CheckInConfig(target_tool="send_message")
        assert config.target_tool == "send_message"
        assert config.max_silent_turns == 3
        assert config.max_interventions == 3
        assert config.cooldown_turns == 1
        assert "{turns_since}" in config.reminder_template
        assert "{tool}" in config.reminder_template

    def test_custom_values(self) -> None:
        config = CheckInConfig(
            target_tool="ping",
            max_silent_turns=5,
            max_interventions=2,
            cooldown_turns=3,
        )
        assert config.target_tool == "ping"
        assert config.max_silent_turns == 5
        assert config.max_interventions == 2
        assert config.cooldown_turns == 3

    def test_max_silent_turns_lower_bound(self) -> None:
        with pytest.raises(Exception):
            CheckInConfig(target_tool="t", max_silent_turns=0)

    def test_max_silent_turns_upper_bound(self) -> None:
        with pytest.raises(Exception):
            CheckInConfig(target_tool="t", max_silent_turns=21)

    def test_max_interventions_zero_allowed(self) -> None:
        config = CheckInConfig(target_tool="t", max_interventions=0)
        assert config.max_interventions == 0

    def test_max_interventions_upper_bound(self) -> None:
        with pytest.raises(Exception):
            CheckInConfig(target_tool="t", max_interventions=21)

    def test_cooldown_zero_allowed(self) -> None:
        config = CheckInConfig(target_tool="t", cooldown_turns=0)
        assert config.cooldown_turns == 0

    def test_cooldown_upper_bound(self) -> None:
        with pytest.raises(Exception):
            CheckInConfig(target_tool="t", cooldown_turns=11)

    def test_template_formatting(self) -> None:
        config = CheckInConfig(target_tool="send_message")
        result = config.reminder_template.format(turns_since=4, tool="send_message")
        assert "4" in result
        assert "send_message" in result


# ---------------------------------------------------------------------------
# TestCheckInDetector
# ---------------------------------------------------------------------------


class TestCheckInDetector:
    """Unit tests for CheckInDetector logic."""

    @pytest.fixture()
    def config(self) -> CheckInConfig:
        return CheckInConfig(
            target_tool="send_message",
            max_silent_turns=3,
            max_interventions=3,
            cooldown_turns=1,
        )

    @pytest.fixture()
    def detector(self, config: CheckInConfig) -> CheckInDetector:
        return CheckInDetector(config)

    def test_initial_state(self, detector: CheckInDetector, config: CheckInConfig) -> None:
        assert detector.turns_since_target == 0
        assert detector.intervention_count == 0
        # Starts at cooldown_turns so first eligible turn can fire immediately.
        assert detector.turns_since_last_intervention == config.cooldown_turns

    def test_record_turn_with_target_tool(self, detector: CheckInDetector) -> None:
        """Using the target tool resets the silent-turn counter."""
        # Increment the counter first so we can verify a reset.
        detector.turns_since_target = 5
        detector.record_turn([_make_ai_with_tool("send_message")])
        assert detector.turns_since_target == 0

    def test_record_turn_without_target_tool(self, detector: CheckInDetector) -> None:
        """Not using the target tool increments the silent-turn counter."""
        detector.record_turn([_make_ai_no_tool()])
        assert detector.turns_since_target == 1
        detector.record_turn([_make_ai_no_tool()])
        assert detector.turns_since_target == 2

    def test_record_turn_other_tool_counts_as_silent(
        self, detector: CheckInDetector
    ) -> None:
        """Calling a different tool does not reset the check-in counter."""
        detector.record_turn([_make_ai_with_tool("do_research")])
        assert detector.turns_since_target == 1

    def test_record_turn_increments_cooldown_counter(
        self, detector: CheckInDetector
    ) -> None:
        initial = detector.turns_since_last_intervention
        detector.record_turn([_make_ai_no_tool()])
        assert detector.turns_since_last_intervention == initial + 1

    def test_should_intervene_after_threshold(
        self, detector: CheckInDetector, config: CheckInConfig
    ) -> None:
        """should_intervene returns True once silent turns >= max_silent_turns."""
        for _ in range(config.max_silent_turns):
            detector.record_turn([_make_ai_no_tool()])
        assert detector.should_intervene()

    def test_should_not_intervene_before_threshold(
        self, detector: CheckInDetector, config: CheckInConfig
    ) -> None:
        """should_intervene is False when still below the silent-turn threshold."""
        for _ in range(config.max_silent_turns - 1):
            detector.record_turn([_make_ai_no_tool()])
        assert not detector.should_intervene()

    def test_should_not_intervene_during_cooldown(
        self, config: CheckInConfig
    ) -> None:
        """An intervention resets the cooldown; should_intervene returns False immediately."""
        config = CheckInConfig(
            target_tool="send_message",
            max_silent_turns=1,
            max_interventions=5,
            cooldown_turns=2,
        )
        detector = CheckInDetector(config)
        # Trigger threshold.
        detector.record_turn([_make_ai_no_tool()])
        assert detector.should_intervene()

        detector.record_intervention()
        # Cooldown resets to 0; cooldown_turns is 2, so not ready yet.
        assert not detector.should_intervene()

    def test_max_interventions_respected(
        self, detector: CheckInDetector, config: CheckInConfig
    ) -> None:
        """should_intervene returns False once max_interventions is exhausted."""
        detector.intervention_count = config.max_interventions
        detector.turns_since_target = config.max_silent_turns
        assert not detector.should_intervene()

    def test_build_reminder(self, detector: CheckInDetector, config: CheckInConfig) -> None:
        detector.turns_since_target = 4
        reminder = detector.build_reminder()
        assert "4" in reminder
        assert config.target_tool in reminder

    def test_record_intervention(self, detector: CheckInDetector) -> None:
        detector.record_intervention()
        assert detector.intervention_count == 1
        assert detector.turns_since_last_intervention == 0

    def test_reset(self, detector: CheckInDetector, config: CheckInConfig) -> None:
        detector.turns_since_target = 10
        detector.intervention_count = 2
        detector.turns_since_last_intervention = 0

        detector.reset()

        assert detector.turns_since_target == 0
        assert detector.intervention_count == 0
        assert detector.turns_since_last_intervention == config.cooldown_turns


# ---------------------------------------------------------------------------
# TestCheckInMiddleware
# ---------------------------------------------------------------------------


class TestCheckInMiddleware:
    """Integration tests for CheckInMiddleware.wrap_model_call."""

    @pytest.fixture()
    def config(self) -> CheckInConfig:
        return CheckInConfig(
            target_tool="send_message",
            max_silent_turns=2,
            max_interventions=3,
            cooldown_turns=1,
        )

    @pytest.fixture()
    def middleware(self, config: CheckInConfig) -> CheckInMiddleware:
        return CheckInMiddleware(config)

    def test_construction(
        self, middleware: CheckInMiddleware, config: CheckInConfig
    ) -> None:
        assert middleware.detector is not None
        assert middleware.detector.turns_since_target == 0

    def test_detector_property_exposed(self, middleware: CheckInMiddleware) -> None:
        assert isinstance(middleware.detector, CheckInDetector)

    def test_wrap_model_call_no_injection_when_healthy(
        self, middleware: CheckInMiddleware
    ) -> None:
        """No framework_instruction injected when the agent is within threshold."""
        request = make_mock_request(messages=[HumanMessage(content="hi")])
        handler = make_mock_handler()

        middleware.wrap_model_call(request, handler)

        # override should not have been called (no intervention needed)
        request.override.assert_not_called()

    def test_wrap_model_call_injects_when_drifting(
        self, middleware: CheckInMiddleware, config: CheckInConfig
    ) -> None:
        """A framework_instruction is injected once silent turns exceed threshold."""
        # Simulate enough silent turns to cross the threshold.
        for _ in range(config.max_silent_turns):
            middleware.detector.record_turn([_make_ai_no_tool()])

        assert middleware.detector.should_intervene()

        request = make_mock_request(messages=[HumanMessage(content="continue")])
        handler = make_mock_handler()

        middleware.wrap_model_call(request, handler)

        # override must have been called and the HumanMessage should now
        # contain a <framework_instruction> block.
        request.override.assert_called_once()
        injected_messages = request.override.call_args.kwargs["messages"]
        human_msg = next(m for m in injected_messages if isinstance(m, HumanMessage))
        assert "<framework_instruction>" in human_msg.content
        assert "send_message" in human_msg.content

    def test_wrap_model_call_records_turn(
        self, middleware: CheckInMiddleware
    ) -> None:
        """record_turn is called on the response after each handler invocation."""
        request = make_mock_request(messages=[HumanMessage(content="hi")])
        response_msg = _make_ai_with_tool("send_message")
        handler = make_mock_handler(response_messages=[response_msg])

        middleware.wrap_model_call(request, handler)

        # The target tool was in the response, so the counter should have reset.
        assert middleware.detector.turns_since_target == 0

    def test_wrap_model_call_returns_extended_response(
        self, middleware: CheckInMiddleware
    ) -> None:
        """wrap_model_call returns an ExtendedModelResponse with state updates."""
        from langchain.agents.middleware import ExtendedModelResponse

        request = make_mock_request(messages=[HumanMessage(content="hi")])
        handler = make_mock_handler()

        result = middleware.wrap_model_call(request, handler)

        assert isinstance(result, ExtendedModelResponse)

    def test_reset_recreates_detector(self, middleware: CheckInMiddleware) -> None:
        middleware.detector.turns_since_target = 7
        middleware.detector.intervention_count = 2
        middleware.reset()
        assert middleware.detector.turns_since_target == 0
        assert middleware.detector.intervention_count == 0


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end scenario: normal usage, drift, intervention, recovery."""

    def test_full_check_in_scenario(self) -> None:
        config = CheckInConfig(
            target_tool="send_message",
            max_silent_turns=3,
            max_interventions=2,
            cooldown_turns=1,
        )
        detector = CheckInDetector(config)

        # Phase 1 — healthy usage: agent sends a status update each turn.
        for _ in range(3):
            detector.record_turn([_make_ai_with_tool("send_message")])

        assert detector.turns_since_target == 0
        assert not detector.should_intervene()

        # Phase 2 — drift: agent stops sending updates.
        for _ in range(3):
            detector.record_turn([_make_ai_no_tool()])

        assert detector.turns_since_target == 3
        assert detector.should_intervene()

        # Phase 3 — first intervention.
        reminder = detector.build_reminder()
        assert "send_message" in reminder
        detector.record_intervention()
        assert detector.intervention_count == 1
        assert detector.turns_since_last_intervention == 0

        # Cooldown in effect immediately after intervention.
        assert not detector.should_intervene()

        # Phase 4 — agent keeps drifting for one more turn (cooldown resets).
        detector.record_turn([_make_ai_no_tool()])
        # turns_since_target is now 4 (still above threshold), cooldown elapsed.
        assert detector.should_intervene()

        # Phase 5 — recovery: agent sends a status update.
        detector.record_turn([_make_ai_with_tool("send_message")])
        assert detector.turns_since_target == 0
        assert not detector.should_intervene()

    def test_max_interventions_halts_reminders(self) -> None:
        config = CheckInConfig(
            target_tool="send_message",
            max_silent_turns=1,
            max_interventions=1,
            cooldown_turns=0,
        )
        detector = CheckInDetector(config)

        # Cross threshold.
        detector.record_turn([_make_ai_no_tool()])
        assert detector.should_intervene()

        # Fire the single allowed intervention.
        detector.record_intervention()

        # Now max_interventions is exhausted — no more reminders.
        detector.record_turn([_make_ai_no_tool()])
        assert not detector.should_intervene()

    def test_middleware_state_updates_flow_through(self) -> None:
        """State fields are written into the ExtendedModelResponse command."""
        from langchain.agents.middleware import ExtendedModelResponse

        config = CheckInConfig(
            target_tool="send_message",
            max_silent_turns=10,  # High threshold so no intervention fires.
            max_interventions=3,
            cooldown_turns=1,
        )
        middleware = CheckInMiddleware(config)

        request = make_mock_request(messages=[HumanMessage(content="go")])
        handler = make_mock_handler(response_messages=[_make_ai_no_tool()])

        result = middleware.wrap_model_call(request, handler)

        assert isinstance(result, ExtendedModelResponse)
        update = result.command.update  # type: ignore[union-attr]
        assert "turns_since_check_in" in update
        assert "intervention_count" in update
        assert update["turns_since_check_in"] == 1
        assert update["intervention_count"] == 0
