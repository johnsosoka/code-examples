"""Tool-triggered phase transition detector.

Pure lookup logic: no heuristics, no content scanning, no pattern sets.
"""

from __future__ import annotations

from bsm_examples.phase_steering.models import Phase, PhaseConfig


class ToolTriggeredDetector:
    """Determines phase transitions based on which tool was just called.

    This is a stateless lookup: given the current phase and a tool name,
    it consults the ``PhaseConfig`` transition rules and returns the target
    phase if one matches, or ``None`` if the tool does not trigger any
    transition from the current phase.

    Example:
        >>> detector = ToolTriggeredDetector(config)
        >>> next_phase = detector.check_tool_transition(Phase.DISCOVERY, "note_preferences")
        >>> next_phase
        <Phase.CURATION: 2>
    """

    def __init__(self, config: PhaseConfig) -> None:
        """Initialize with a phase configuration.

        Args:
            config: The workflow's phase configuration containing transition rules.
        """
        self.config = config

    def check_tool_transition(
        self, current_phase: Phase, tool_name: str
    ) -> Phase | None:
        """Check whether a tool call triggers a phase transition.

        Args:
            current_phase: The phase the agent is currently in.
            tool_name: The name of the tool that was just executed.

        Returns:
            The target ``Phase`` if a matching transition rule exists,
            or ``None`` if the tool call does not trigger any transition.
        """
        rule = self.config.get_transition_for_tool(current_phase, tool_name)
        return rule.to_phase if rule else None
