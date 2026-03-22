"""Phase steering models: enums, behaviors, transitions, and configuration.

Merges the old phase.py and config.py into a single, cohesive module.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Self

from pydantic import BaseModel, Field, field_validator


class Phase(Enum):
    """Workflow phases for the personalization steering workflow."""

    DISCOVERY = auto()
    CURATION = auto()
    RECOMMENDATION = auto()
    COMPLETE = auto()

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Parse a Phase from a string name (case-insensitive).

        Args:
            value: Phase name string, e.g. "discovery" or "DISCOVERY".

        Returns:
            Matching Phase enum member.

        Raises:
            ValueError: If the string does not match any Phase name.
        """
        try:
            return cls[value.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown phase: {value}") from e

    @property
    def display_name(self) -> str:
        """Human-readable title-cased phase name."""
        return self.name.title()

    @property
    def order(self) -> int:
        """Zero-based index of this phase in the canonical Phase ordering."""
        return list(Phase).index(self)


class PhaseBehavior(BaseModel):
    """Behavior configuration for a single workflow phase.

    Attributes:
        system_prompt_addition: Guidance text injected into the system prompt
            while the agent is in this phase.
        allowed_tools: Tools the agent may call in this phase. An empty list
            means all tools are permitted.
        completion_criteria: Optional human-readable description of what
            constitutes completion of this phase.
        max_iterations: Maximum model calls allowed while in this phase before
            the workflow should forcibly advance.
    """

    system_prompt_addition: str
    allowed_tools: list[str] = Field(default_factory=list)
    completion_criteria: str = ""
    max_iterations: int = Field(default=10, ge=1, le=50)


class PhaseTransitionRule(BaseModel):
    """Defines a tool-triggered phase transition.

    When the agent calls ``trigger_tool`` while in ``from_phase``, the
    middleware transitions to ``to_phase``.

    Attributes:
        from_phase: Phase in which this rule is active.
        to_phase: Phase to transition into when the trigger fires.
        trigger_tool: Name of the tool call that activates this rule.
        description: Optional human-readable description for documentation.
    """

    from_phase: Phase
    to_phase: Phase
    trigger_tool: str = Field(description="Tool call that triggers this transition")
    description: str = ""


class PhaseConfig(BaseModel):
    """Top-level configuration for a phase-steered workflow.

    Attributes:
        phases: Mapping from each Phase to its behavior configuration.
        transitions: Ordered list of transition rules evaluated after each
            tool call.
        initial_phase: Phase the agent starts in. Defaults to DISCOVERY.
    """

    phases: dict[Phase, PhaseBehavior]
    transitions: list[PhaseTransitionRule] = Field(default_factory=list)
    initial_phase: Phase = Phase.DISCOVERY

    @field_validator("phases")
    @classmethod
    def validate_phases_not_empty(cls, v: dict[Phase, PhaseBehavior]) -> dict[Phase, PhaseBehavior]:
        """Ensure at least one phase is defined."""
        if not v:
            raise ValueError("At least one phase must be defined")
        return v

    def get_transition_for_tool(
        self, current_phase: Phase, tool_name: str
    ) -> PhaseTransitionRule | None:
        """Look up the transition rule triggered by a specific tool in the current phase.

        Args:
            current_phase: The phase the agent is currently in.
            tool_name: The name of the tool that was just called.

        Returns:
            The matching ``PhaseTransitionRule``, or ``None`` if no rule matches.
        """
        for rule in self.transitions:
            if rule.from_phase == current_phase and rule.trigger_tool == tool_name:
                return rule
        return None
