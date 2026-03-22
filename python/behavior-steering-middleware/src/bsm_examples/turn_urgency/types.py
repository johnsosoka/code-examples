"""Type definitions for turn-based urgency middleware.

This module contains enums and Pydantic models for configuring and
controlling urgency-based behavior steering.
"""

from __future__ import annotations

from enum import Enum, auto

from langchain.agents.middleware import AgentState
from pydantic import BaseModel, Field, model_validator
from typing_extensions import NotRequired


class UrgencyLevel(Enum):
    """Four distinct urgency modes representing the agent's behavioral state.

    The progression is designed to be gradual—each level removes some flexibility
    while maintaining core helpfulness until the terminal state.
    """

    EXPLORATION = auto()
    """Full turn budget available. Agent is encouraged to:
    - Ask clarifying questions
    - Explore multiple solution approaches
    - Provide comprehensive context and background
    - Consider edge cases and alternatives
    """

    BALANCED = auto()
    """Moderate budget remaining. Agent should:
    - Answer directly while maintaining helpfulness
    - Limit follow-up questions to critical items only
    - Provide concise but complete responses
    - Focus on the most likely solution paths
    """

    EFFICIENCY = auto()
    """Low budget remaining. Agent must:
    - Prioritize delivering answers over asking questions
    - Make reasonable assumptions rather than seeking clarification
    - Provide minimal viable solutions
    - Avoid tangential information
    """

    TERMINAL = auto()
    """Final turn available. Agent is instructed to:
    - Deliver any workable solution immediately
    - No clarifying questions—use best judgment
    - Summarize current progress if task incomplete
    - Provide clear next steps for continuation if needed
    """


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
    def validate_order(self) -> "UrgencyThresholds":
        """Ensure thresholds are in descending order."""
        if self.balanced_max >= self.exploration_max:
            raise ValueError("balanced_max must be less than exploration_max")
        if self.efficiency_max >= self.balanced_max:
            raise ValueError("efficiency_max must be less than balanced_max")
        return self


class UrgencyPrompts(BaseModel):
    """System prompt modifiers for each urgency level."""

    exploration: str = Field(
        default=(
            "You have ample turns remaining. Feel free to ask clarifying questions, "
            "explore multiple approaches, and provide comprehensive context. "
            "Consider edge cases and discuss trade-offs openly."
        ),
    )
    balanced: str = Field(
        default=(
            "You have moderate turns remaining. Provide direct answers with minimal "
            "follow-up questions. Focus on the most likely solution paths while "
            "maintaining helpfulness and accuracy."
        ),
    )
    efficiency: str = Field(
        default=(
            "You have limited turns remaining. Prioritize delivering solutions over "
            "asking questions. Make reasonable assumptions and provide minimal viable "
            "answers. Avoid tangential information."
        ),
    )
    terminal: str = Field(
        default=(
            "FINAL TURN: Deliver your best solution immediately without asking questions. "
            "If the task is incomplete, summarize progress and provide clear next steps. "
            "Do not ask for clarification—use your best judgment."
        ),
    )


class TurnUrgencyConfig(BaseModel):
    """Configuration for turn-based urgency middleware.

    All fields have sensible defaults but can be customized for specific use cases.

    Example:
        >>> config = TurnUrgencyConfig(max_turns=10)
        >>> config = TurnUrgencyConfig(
        ...     max_turns=5,
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

    @model_validator(mode="after")
    def validate_counting_mode(self) -> "TurnUrgencyConfig":
        """Ensure only one counting mode is enabled."""
        if self.count_human_turns_only and self.count_agent_turns_only:
            raise ValueError(
                "Cannot enable both count_human_turns_only and count_agent_turns_only"
            )
        return self


class UrgencyAgentState(AgentState):
    """Agent state schema with urgency tracking fields.

    Extends AgentState for use with create_agent and AgentMiddleware.
    All urgency fields are optional (NotRequired) since they are populated
    progressively as the middleware processes model calls.
    """

    urgency_level: NotRequired[str]
    turns_used: NotRequired[int]
    turns_remaining: NotRequired[int]
    is_terminal: NotRequired[bool]
