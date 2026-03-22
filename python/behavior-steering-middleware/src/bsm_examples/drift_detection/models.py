"""Configuration models for the Worker Check-In Monitor middleware."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CheckInConfig(BaseModel):
    """Configuration for the check-in monitoring middleware.

    Monitors a target tool (e.g., ``send_message``) and injects a reminder
    into the message stream when the agent has not used it in N turns.

    Example:
        >>> config = CheckInConfig(target_tool="send_message", max_silent_turns=2)
    """

    target_tool: str = Field(description="Tool name to monitor for check-in activity")

    max_silent_turns: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of turns without the target tool before an intervention fires",
    )

    max_interventions: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Maximum number of reminder injections per conversation",
    )

    cooldown_turns: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Minimum turns between consecutive interventions",
    )

    reminder_template: str = Field(
        default=(
            "You have not sent a status update in {turns_since} turns. "
            "Please use the `{tool}` tool to report your progress."
        ),
        description="Template for the reminder message injected as a <system> tag",
    )
