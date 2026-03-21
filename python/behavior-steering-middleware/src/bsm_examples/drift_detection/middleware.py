"""Behavior Drift Detection middleware implementation.

This module provides the core middleware classes for detecting and correcting
behavior drift in LLM-based agents.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToolUsagePattern(BaseModel):
    """
    Expected pattern of tool usage for drift detection.

    Defines which tools should be used, expected call frequency,
    and acceptable variance bounds for each tool.
    """

    name: str = Field(description="Human-readable name for this pattern")

    # Expected tools and their baseline frequencies (0.0-1.0)
    # Frequency = expected proportion of turns that invoke this tool
    expected_tools: dict[str, float] = Field(
        default_factory=dict,
        description="Tool name -> expected call frequency (0.0-1.0)",
    )

    # Minimum acceptable frequency (allows for "at least this often" constraints)
    minimum_frequencies: dict[str, float] = Field(
        default_factory=dict,
        description="Tool name -> minimum acceptable frequency",
    )

    # Tools that MUST be called at least once in the conversation
    required_tools: list[str] = Field(
        default_factory=list,
        description="Tools that must be called at least once",
    )

    # Tools that should NEVER be called (for safety/policy enforcement)
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="Tools that must never be invoked",
    )

    @field_validator("expected_tools", "minimum_frequencies")
    @classmethod
    def validate_frequencies(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure all frequencies are in valid range [0.0, 1.0]."""
        for tool, freq in v.items():
            if not 0.0 <= freq <= 1.0:
                raise ValueError(
                    f"Frequency for {tool} must be in [0.0, 1.0], got {freq}"
                )
        return v

    def get_expected_frequency(self, tool_name: str) -> float:
        """Get expected frequency for a tool, defaulting to 0.0."""
        return self.expected_tools.get(tool_name, 0.0)

    def get_minimum_frequency(self, tool_name: str) -> float | None:
        """Get minimum acceptable frequency for a tool."""
        return self.minimum_frequencies.get(tool_name)


class DriftDetectionConfig(BaseModel):
    """
    Configuration for drift detection middleware.

    Controls sensitivity, history window, and intervention thresholds.
    """

    # History window configuration
    history_window_size: int = Field(
        default=10,
        ge=3,
        le=100,
        description="Number of recent turns to include in drift calculation",
    )

    # Weighting: how much to favor recent vs older history
    # 1.0 = linear (equal weight), >1.0 = recency-biased, <1.0 = history-biased
    recency_bias: float = Field(
        default=1.5,
        ge=0.1,
        le=5.0,
        description="Exponent for recency weighting (1.5 = favor recent)",
    )

    # Drift score thresholds
    warning_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Drift score at which to log warnings",
    )

    intervention_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Drift score at which to inject corrective reminders",
    )

    # Intervention configuration
    max_interventions_per_conversation: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Maximum corrective reminders per conversation",
    )

    cooldown_turns: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Turns to wait between interventions",
    )

    # Reminder message template
    reminder_template: str = Field(
        default=(
            "[SYSTEM REMINDER] You appear to be responding without using tools. "
            "For this task, you should use: {tools}. "
            "Please invoke the appropriate tool before responding."
        ),
        description="Template for corrective reminder messages",
    )

    # Whether to include tool usage statistics in reminder
    include_statistics: bool = Field(
        default=True,
        description="Include observed vs expected frequencies in reminder",
    )

    @field_validator("intervention_threshold")
    @classmethod
    def validate_thresholds(cls, v: float, info: Any) -> float:
        """Ensure intervention threshold is >= warning threshold."""
        warning = info.data.get("warning_threshold", 0.3)
        if v < warning:
            raise ValueError(
                f"intervention_threshold ({v}) must be >= warning_threshold ({warning})"
            )
        return v


@dataclass
class TurnRecord:
    """Record of a single turn's tool usage."""

    turn_number: int
    tools_called: list[str] = field(default_factory=list)
    was_tool_response: bool = False  # Was this turn a response to a tool result?


class DriftDetector:
    """
    Detects deviation from expected tool usage patterns.

    Maintains a sliding window of recent turns, calculates weighted
    drift scores, and determines when intervention is needed.
    """

    def __init__(
        self,
        pattern: ToolUsagePattern,
        config: DriftDetectionConfig,
    ) -> None:
        """Initialize the drift detector.

        Args:
            pattern: Expected tool usage pattern
            config: Drift detection configuration
        """
        self.pattern = pattern
        self.config = config
        self.history: deque[TurnRecord] = deque(maxlen=config.history_window_size)
        self.turn_count = 0
        self.intervention_count = 0
        self.turns_since_last_intervention = config.cooldown_turns  # Ready immediately

    def record_turn(self, messages: list[BaseMessage]) -> None:
        """
        Record tool usage from a completed turn.

        Analyzes messages to identify tool calls and responses.

        Args:
            messages: List of messages from the turn
        """
        self.turn_count += 1
        tools_called: list[str] = []
        was_tool_response = False

        for msg in messages:
            # Detect tool calls in AIMessage
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    tool_name = tc.get("name") or tc.get("function", {}).get("name")
                    if tool_name:
                        tools_called.append(tool_name)
                        logger.debug(
                            f"Turn {self.turn_count}: detected tool call to {tool_name}"
                        )

            # Detect tool responses
            if isinstance(msg, ToolMessage):
                was_tool_response = True

        record = TurnRecord(
            turn_number=self.turn_count,
            tools_called=tools_called,
            was_tool_response=was_tool_response,
        )
        self.history.append(record)
        self.turns_since_last_intervention += 1

        # Log current state
        drift_score = self.calculate_drift_score()
        logger.debug(
            f"Turn {self.turn_count} recorded: tools={tools_called}, "
            f"drift_score={drift_score:.3f}"
        )

    def calculate_drift_score(self) -> float:
        """
        Calculate weighted drift score from 0.0 (no drift) to 1.0 (max drift).

        Algorithm:
        1. Calculate weighted observed frequencies from history
        2. Compare against expected frequencies
        3. Combine deviations into aggregate drift score

        Returns:
            Drift score between 0.0 and 1.0
        """
        if not self.history:
            return 0.0

        # Calculate weighted observed frequencies
        weighted_observed = self._calculate_weighted_frequencies()

        # Calculate drift for each expected tool
        tool_drifts: list[float] = []

        for tool_name, expected_freq in self.pattern.expected_tools.items():
            observed_freq = weighted_observed.get(tool_name, 0.0)

            # Calculate raw deviation
            if expected_freq > 0:
                # Under-use is the primary concern for drift
                deviation = max(0.0, expected_freq - observed_freq) / expected_freq
            else:
                # If not expected but observed, that's also drift
                deviation = min(1.0, observed_freq * 2)  # Scale up unexpected usage

            # Apply minimum frequency constraint if specified
            min_freq = self.pattern.get_minimum_frequency(tool_name)
            if min_freq is not None and observed_freq < min_freq:
                # Extra penalty for violating minimum
                min_violation = (
                    (min_freq - observed_freq) / min_freq if min_freq > 0 else 0.0
                )
                deviation = max(deviation, min_violation)

            tool_drifts.append(deviation)
            logger.debug(
                f"Tool {tool_name}: expected={expected_freq:.3f}, "
                f"observed={observed_freq:.3f}, deviation={deviation:.3f}"
            )

        # Check required tools (must be called at least once in entire conversation)
        for tool_name in self.pattern.required_tools:
            ever_called = any(
                tool_name in record.tools_called for record in self.history
            )
            if not ever_called and self.turn_count >= 3:
                # Penalty increases as conversation progresses
                required_penalty = min(1.0, self.turn_count / 10)
                tool_drifts.append(required_penalty)

        # Check forbidden tools (immediate max drift if called)
        for record in self.history:
            for tool_name in record.tools_called:
                if tool_name in self.pattern.forbidden_tools:
                    logger.warning(f"Forbidden tool {tool_name} was called!")
                    return 1.0

        if not tool_drifts:
            return 0.0

        # Aggregate: use max drift (worst offender) with mean as secondary factor
        max_drift = max(tool_drifts)
        mean_drift = sum(tool_drifts) / len(tool_drifts)

        # Weighted combination: max is more important but mean matters too
        aggregate_drift = 0.7 * max_drift + 0.3 * mean_drift

        return min(1.0, max(0.0, aggregate_drift))

    def _calculate_weighted_frequencies(self) -> dict[str, float]:
        """
        Calculate frequency of each tool with recency weighting.

        Returns:
            Dict of tool_name -> weighted frequency [0.0, 1.0]
        """
        if not self.history:
            return {}

        # Calculate weights based on position (newer = higher weight)
        n = len(self.history)
        weights: list[float] = []

        for i, _ in enumerate(self.history):
            # Position 0 = oldest, position n-1 = newest
            # Normalize to [0, 1] then apply recency bias exponent
            normalized_pos = i / (n - 1) if n > 1 else 1.0
            weight = normalized_pos ** self.config.recency_bias
            weights.append(weight)

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / n] * n
        else:
            weights = [w / total_weight for w in weights]

        # Calculate weighted tool usage
        tool_weights: dict[str, float] = {}

        for record, weight in zip(self.history, weights):
            for tool in record.tools_called:
                tool_weights[tool] = tool_weights.get(tool, 0.0) + weight

        # Convert to frequencies
        # The weighted sum represents the proportion of "weighted turns" that used this tool
        # Since weights sum to 1, max possible for any tool is 1 (if called in all turns)
        frequencies: dict[str, float] = {}
        for tool, total_weight in tool_weights.items():
            # total_weight is already normalized (sum of weights = 1)
            # If a tool is called in every turn, total_weight would be 1.0
            frequencies[tool] = min(1.0, total_weight)

        return frequencies

    def should_intervene(self) -> bool:
        """
        Determine if corrective intervention is needed.

        Checks:
        1. Drift score exceeds intervention threshold
        2. Haven't exceeded max interventions
        3. Cooldown period has passed

        Returns:
            True if intervention should be performed
        """
        if (
            self.intervention_count
            >= self.config.max_interventions_per_conversation
        ):
            return False

        if self.turns_since_last_intervention < self.config.cooldown_turns:
            return False

        drift_score = self.calculate_drift_score()

        if drift_score >= self.config.intervention_threshold:
            return True

        return False

    def record_intervention(self) -> None:
        """Record that an intervention was performed."""
        self.intervention_count += 1
        self.turns_since_last_intervention = 0
        logger.info(f"Intervention #{self.intervention_count} performed")

    def get_drift_report(self) -> dict[str, Any]:
        """Generate a diagnostic report of current drift state."""
        weighted_observed = self._calculate_weighted_frequencies()

        return {
            "turn_count": self.turn_count,
            "history_size": len(self.history),
            "drift_score": self.calculate_drift_score(),
            "intervention_count": self.intervention_count,
            "tool_comparison": {
                tool: {
                    "expected": self.pattern.get_expected_frequency(tool),
                    "observed": weighted_observed.get(tool, 0.0),
                    "minimum": self.pattern.get_minimum_frequency(tool),
                }
                for tool in self.pattern.expected_tools.keys()
            },
        }


class DriftSteeringMiddleware(AgentMiddleware):
    """
    Middleware that monitors tool usage patterns and steers agent behavior
    back toward expected patterns when drift is detected.

    This middleware wraps model calls to:
    1. Record tool usage from each turn
    2. Calculate drift scores based on recent history
    3. Inject corrective reminders when drift exceeds threshold

    Example:
        pattern = ToolUsagePattern(
            name="calculator_agent",
            expected_tools={"calculator": 0.8, "wikipedia": 0.2},
            required_tools=["calculator"],
        )
        config = DriftDetectionConfig(intervention_threshold=0.5)

        middleware = DriftSteeringMiddleware(pattern, config)

        agent = create_agent(
            model=model,
            tools=[calculator_tool, wikipedia_tool],
            middleware=[middleware],
        )
    """

    def __init__(
        self,
        pattern: ToolUsagePattern,
        config: DriftDetectionConfig | None = None,
    ) -> None:
        """Initialize the drift steering middleware.

        Args:
            pattern: Expected tool usage pattern
            config: Optional drift detection configuration (uses defaults if None)
        """
        super().__init__()
        self.pattern = pattern
        self.config = config or DriftDetectionConfig()
        self.detector = DriftDetector(pattern, self.config)
        self._intervention_messages: list[SystemMessage] = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """
        Wrap model call to inject drift corrections before inference.

        Flow:
        1. Check if intervention is needed based on current drift
        2. If so, prepend corrective reminder to messages
        3. Call model with (possibly modified) request
        4. Record tool usage from response for next turn's analysis

        Args:
            request: The model request
            handler: Function to call the model

        Returns:
            ModelResponse from the handler
        """
        # Check for drift and prepare intervention
        modified_messages = list(request.messages)

        if self.detector.should_intervene():
            reminder = self._generate_reminder()
            modified_messages.insert(0, SystemMessage(content=reminder))
            self.detector.record_intervention()
            logger.warning(
                f"Drift detected (score={self.detector.calculate_drift_score():.3f}). "
                f"Injecting corrective reminder."
            )

        # Create modified request if needed
        if len(modified_messages) != len(request.messages):
            request = request.override(messages=modified_messages)

        # Call the model
        response = handler(request)

        # Record tool usage from this turn
        if response.result:
            all_messages = list(request.messages) + list(response.result)
            self.detector.record_turn(all_messages)

            # Log drift status
            drift_score = self.detector.calculate_drift_score()
            if drift_score >= self.config.warning_threshold:
                logger.info(f"Current drift score: {drift_score:.3f}")

        return response

    def _generate_reminder(self) -> str:
        """Generate a contextual reminder message based on current drift."""
        # Identify under-used tools
        weighted_observed = self.detector._calculate_weighted_frequencies()
        underused_tools: list[str] = []

        for tool, expected in self.pattern.expected_tools.items():
            observed = weighted_observed.get(tool, 0.0)
            if observed < expected * 0.5:  # Less than half expected
                underused_tools.append(tool)

        # Identify missing required tools
        missing_required: list[str] = []
        for tool in self.pattern.required_tools:
            ever_called = any(
                tool in record.tools_called for record in self.detector.history
            )
            if not ever_called:
                missing_required.append(tool)

        # Build tool list for reminder
        tools_to_mention = underused_tools or list(
            self.pattern.expected_tools.keys()
        )
        tools_str = ", ".join(f"`{t}`" for t in tools_to_mention[:3])

        # Generate reminder
        reminder = self.config.reminder_template.format(tools=tools_str)

        # Add statistics if configured
        if self.config.include_statistics:
            report = self.detector.get_drift_report()
            stats = f" [Current drift score: {report['drift_score']:.2f}]"
            reminder = reminder.rstrip() + stats

        return reminder

    def get_drift_report(self) -> dict[str, Any]:
        """Get current drift detection report."""
        return self.detector.get_drift_report()

    def reset(self) -> None:
        """Reset the middleware state for a new conversation."""
        self.detector = DriftDetector(self.pattern, self.config)
        self._intervention_messages = []
        logger.debug("DriftSteeringMiddleware reset for new conversation")
