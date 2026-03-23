"""Pydantic models and enums for turn-based urgency middleware.

Contains the configuration models, urgency level enum, and prompt defaults
used across the turn urgency package.
"""

from __future__ import annotations

from enum import Enum, auto

from pydantic import BaseModel, Field, model_validator


class UrgencyLevel(Enum):
    """Four distinct urgency modes representing the agent's behavioral state.

    The progression is designed to be gradual — each level removes some flexibility
    while maintaining core helpfulness until the terminal state.
    """

    EXPLORATION = auto()
    """Full turn budget available. Agent is encouraged to ask clarifying questions,
    explore multiple solution approaches, and gather comprehensive information."""

    BALANCED = auto()
    """Moderate budget remaining. Agent should start narrowing options and limit
    follow-up questions to critical items only."""

    EFFICIENCY = auto()
    """Low budget remaining. Agent must work toward a recommendation and avoid
    unnecessary clarifying questions."""

    TERMINAL = auto()
    """Final turn available. Agent must use the completion tool immediately
    and deliver a recommendation based on available information."""


class UrgencyThresholds(BaseModel):
    """Threshold definitions for urgency level transitions."""

    exploration_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum ratio for EXPLORATION level (default: 100%)",
    )
    balanced_max: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Maximum ratio for BALANCED level (default: 60%)",
    )
    efficiency_max: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum ratio for EFFICIENCY level (default: 30%)",
    )
    # TERMINAL is implicit: below efficiency_max

    @model_validator(mode="after")
    def validate_order(self) -> UrgencyThresholds:
        """Ensure thresholds are in descending order."""
        if self.balanced_max >= self.exploration_max:
            raise ValueError("balanced_max must be less than exploration_max")
        if self.efficiency_max >= self.balanced_max:
            raise ValueError("efficiency_max must be less than balanced_max")
        return self


class UrgencyPrompts(BaseModel):
    """System prompt modifiers for each urgency level.

    Defaults are tuned for a shopping recommendation scenario where the agent
    must call `recommend_product` before the conversation ends.
    """

    exploration: str = Field(
        default=(
            "Feel free to ask about the customer's preferences, budget, and requirements. "
            "Explore multiple product categories and options."
        ),
    )
    balanced: str = Field(
        default=(
            "Start narrowing down options. Ask only critical clarifying questions. "
            "Begin considering which products best match their needs."
        ),
    )
    efficiency: str = Field(
        default=(
            "Time to work toward a recommendation. If you have enough information, "
            "use the `recommend_product` tool to make your suggestion."
        ),
    )
    terminal: str = Field(
        default=(
            "FINAL TURN: Use the `recommend_product` tool NOW to give the customer a "
            "recommendation. Do not ask any more questions — recommend your best option "
            "based on what you know."
        ),
    )


class TurnUrgencyConfig(BaseModel):
    """Configuration for turn-based urgency middleware.

    All fields have sensible defaults but can be customized for specific use cases.

    Example:
        >>> config = TurnUrgencyConfig(max_turns=10)
        >>> config = TurnUrgencyConfig(
        ...     max_turns=5,
        ...     required_completion_tool="recommend_product",
        ...     thresholds=UrgencyThresholds(balanced_max=0.5, efficiency_max=0.2),
        ... )
    """

    # Budget settings
    max_turns: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of turns allowed for this conversation",
    )

    # Threshold configuration
    thresholds: UrgencyThresholds = Field(
        default_factory=UrgencyThresholds,
        description="Ratio thresholds for urgency level transitions",
    )

    # Prompt configuration
    prompts: UrgencyPrompts = Field(
        default_factory=UrgencyPrompts,
        description="System prompt modifiers for each urgency level",
    )

    # Counting behavior
    count_human_turns_only: bool = Field(
        default=False,
        description="If True, only count human messages toward budget",
    )
    count_agent_turns_only: bool = Field(
        default=False,
        description="If True, only count agent messages toward budget",
    )

    # Completion enforcement
    required_completion_tool: str | None = Field(
        default=None,
        description="Tool that must be called by the final turn",
    )

    @model_validator(mode="after")
    def validate_counting_mode(self) -> TurnUrgencyConfig:
        """Ensure only one counting mode is enabled."""
        if self.count_human_turns_only and self.count_agent_turns_only:
            raise ValueError(
                "Cannot enable both count_human_turns_only and count_agent_turns_only"
            )
        return self
