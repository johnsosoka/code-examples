"""Configuration models for phase-based steering.

This module defines Pydantic models for configuring phase behaviors,
transitions, and overall middleware settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from .phase import Phase


class PhaseBehavior(BaseModel):
    """Behavioral guidance configuration for a specific phase.

    Defines how the agent should behave when in this phase, including
    system prompt additions, allowed tools, and completion criteria.

    Attributes:
        system_prompt_addition: Additional instructions injected during this phase.
        allowed_tools: Tool names permitted in this phase (empty = all allowed).
        completion_criteria: Description of what marks this phase as complete.
        max_iterations: Maximum iterations before forcing phase transition.

    Example:
        >>> behavior = PhaseBehavior(
        ...     system_prompt_addition="Focus on gathering data.",
        ...     allowed_tools=["search", "fetch"],
        ...     max_iterations=5,
        ... )
    """

    system_prompt_addition: str = Field(
        description="Additional instructions injected during this phase"
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Tool names permitted in this phase (empty = all allowed)",
    )
    completion_criteria: str = Field(
        default="",
        description="Description of what marks this phase as complete",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum iterations before forcing phase transition",
    )


class PhaseTransition(BaseModel):
    """Defines valid transitions from one phase to another.

    Includes conditions that must be met and optional validation logic.

    Attributes:
        from_phase: The phase being transitioned from.
        to_phase: The phase being transitioned to.
        condition_description: Natural language description of transition condition.
        requires_tool_call: Specific tool call that triggers this transition.
        requires_state_key: State key that must exist to allow transition.

    Example:
        >>> transition = PhaseTransition(
        ...     from_phase=Phase.GATHER,
        ...     to_phase=Phase.ANALYZE,
        ...     condition_description="At least 3 data items collected",
        ... )
    """

    from_phase: Phase
    to_phase: Phase
    condition_description: str = Field(
        description="Natural language description of transition condition"
    )
    requires_tool_call: str | None = Field(
        default=None,
        description="Specific tool call that triggers this transition",
    )
    requires_state_key: str | None = Field(
        default=None,
        description="State key that must exist to allow transition",
    )


class PhaseConfig(BaseModel):
    """Complete configuration for phase-based steering.

    Defines all phases, their behaviors, and valid transitions between them.
    This is the primary configuration object passed to the middleware.

    Attributes:
        phases: Mapping of phases to their behavior configurations.
        transitions: Valid phase transitions and their conditions.
        initial_phase: Starting phase for new workflows.
        enable_auto_transition: Whether to automatically detect and execute transitions.

    Example:
        >>> config = PhaseConfig(
        ...     phases={
        ...         Phase.GATHER: PhaseBehavior(
        ...             system_prompt_addition="Focus on collecting data...",
        ...             max_iterations=5,
        ...         ),
        ...     },
        ...     initial_phase=Phase.GATHER,
        ... )
    """

    phases: dict[Phase, PhaseBehavior] = Field(
        description="Mapping of phases to their behavior configurations"
    )
    transitions: list[PhaseTransition] = Field(
        default_factory=list,
        description="Valid phase transitions and their conditions",
    )
    initial_phase: Phase | None = Field(
        default=None,  # Will be set to Phase.GATHER in model_post_init
        description="Starting phase for new workflows",
    )
    enable_auto_transition: bool = Field(
        default=True,
        description="Whether to automatically detect and execute phase transitions",
    )

    def model_post_init(self, __context: object) -> None:
        """Set default initial_phase after initialization."""
        if self.initial_phase is None:
            from .phase import Phase

            self.initial_phase = Phase.GATHER

    @field_validator("phases")
    @classmethod
    def validate_phases_not_empty(
        cls, v: dict[Phase, PhaseBehavior]
    ) -> dict[Phase, PhaseBehavior]:
        """Ensure at least one phase is defined."""
        if not v:
            raise ValueError("At least one phase must be defined")
        return v

    @field_validator("transitions")
    @classmethod
    def validate_transitions(cls, v: list[PhaseTransition]) -> list[PhaseTransition]:
        """Ensure transitions don't create invalid cycles (optional validation)."""
        # Could implement cycle detection here for stricter validation
        return v

    def get_allowed_transitions(self, phase: Phase) -> list[Phase]:
        """Get list of phases that can be transitioned to from given phase.

        Args:
            phase: The current phase to check transitions from.

        Returns:
            List of phases that can be transitioned to.

        Example:
            >>> config.get_allowed_transitions(Phase.GATHER)
            [<Phase.ANALYZE: 2>]
        """
        return [t.to_phase for t in self.transitions if t.from_phase == phase]

    def is_valid_transition(self, from_phase: Phase, to_phase: Phase) -> bool:
        """Check if a transition between phases is valid.

        Args:
            from_phase: The phase to transition from.
            to_phase: The phase to transition to.

        Returns:
            True if the transition is defined in the configuration.

        Example:
            >>> config.is_valid_transition(Phase.GATHER, Phase.ANALYZE)
            True
            >>> config.is_valid_transition(Phase.GATHER, Phase.COMPLETE)
            False
        """
        return any(
            t.from_phase == from_phase and t.to_phase == to_phase for t in self.transitions
        )
