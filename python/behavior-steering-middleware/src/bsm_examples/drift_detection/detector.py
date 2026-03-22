"""Core detection logic for the Worker Check-In Monitor middleware.

The CheckInDetector is a plain, stateful class with no LangChain
dependencies. It tracks how many turns have elapsed since the monitored
tool was last called and exposes a simple should_intervene / record_turn
interface consumed by the middleware.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage

from bsm_examples.drift_detection.models import CheckInConfig

logger = logging.getLogger(__name__)


class CheckInDetector:
    """Tracks tool-call silence and determines when to intervene.

    State is fully mutable and intended to be reset between conversations.

    Example:
        >>> config = CheckInConfig(target_tool="send_message", max_silent_turns=3)
        >>> detector = CheckInDetector(config)
        >>> detector.record_turn(messages)
        >>> if detector.should_intervene():
        ...     detector.record_intervention()
        ...     print(detector.build_reminder())
    """

    def __init__(self, config: CheckInConfig) -> None:
        """Initialise the detector.

        Args:
            config: Check-in monitoring configuration.
        """
        self._config = config
        self.turns_since_target: int = 0
        self.intervention_count: int = 0
        # Start ready so the first eligible turn can fire immediately.
        self.turns_since_last_intervention: int = config.cooldown_turns

    def record_turn(self, messages: list[BaseMessage]) -> None:
        """Update counters after the model responds.

        Scans AIMessage tool_calls for the target tool. If found, the silent
        turn counter resets to zero. Otherwise it increments. The cooldown
        counter always increments.

        Args:
            messages: Response messages returned by the model this turn.
        """
        target_used = False

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == self._config.target_tool:
                        target_used = True
                        logger.debug("Check-in detected: %s called", self._config.target_tool)
                        break
            if target_used:
                break

        if target_used:
            self.turns_since_target = 0
        else:
            self.turns_since_target += 1

        self.turns_since_last_intervention += 1

        logger.debug(
            "record_turn complete — turns_since_target=%d, cooldown=%d",
            self.turns_since_target,
            self.turns_since_last_intervention,
        )

    def should_intervene(self) -> bool:
        """Return True when all three intervention gates are open.

        Gates:
        1. Silent turns have reached or exceeded max_silent_turns.
        2. Intervention count is below max_interventions.
        3. Cooldown period has elapsed since the last intervention.

        Returns:
            True if a reminder should be injected this turn.
        """
        silent_enough = self.turns_since_target >= self._config.max_silent_turns
        budget_available = self.intervention_count < self._config.max_interventions
        cooldown_elapsed = (
            self.turns_since_last_intervention >= self._config.cooldown_turns
        )
        return silent_enough and budget_available and cooldown_elapsed

    def record_intervention(self) -> None:
        """Record that an intervention was performed this turn."""
        self.intervention_count += 1
        self.turns_since_last_intervention = 0
        logger.info("Check-in intervention #%d fired", self.intervention_count)

    def build_reminder(self) -> str:
        """Build the formatted reminder string from the configured template.

        Returns:
            Human-readable reminder to be wrapped in ``<framework_instruction>`` tags.
        """
        return self._config.reminder_template.format(
            turns_since=self.turns_since_target,
            tool=self._config.target_tool,
        )

    def reset(self) -> None:
        """Reset all counters to their initial state.

        Call between conversations to avoid state leakage.
        """
        self.turns_since_target = 0
        self.intervention_count = 0
        self.turns_since_last_intervention = self._config.cooldown_turns
        logger.debug("CheckInDetector reset")
