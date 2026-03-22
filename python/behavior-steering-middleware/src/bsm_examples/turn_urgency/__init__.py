"""Turn-Based Urgency middleware for LangGraph agents.

This package provides middleware that dynamically adjusts agent behavior
based on remaining turn budget, creating progressive behavior escalation
from exploratory to urgent as turns are consumed.

Example:
    >>> from bsm_examples.turn_urgency import TurnUrgencyConfig, TurnUrgencyMiddleware
    >>> config = TurnUrgencyConfig(max_turns=5, required_completion_tool="recommend_product")
    >>> middleware = TurnUrgencyMiddleware(config)
"""

from __future__ import annotations

from bsm_examples.turn_urgency.calculator import UrgencyCalculator, UrgencyState
from bsm_examples.turn_urgency.middleware import TurnUrgencyMiddleware
from bsm_examples.turn_urgency.models import (
    TurnUrgencyConfig,
    UrgencyLevel,
    UrgencyPrompts,
    UrgencyThresholds,
)
from bsm_examples.turn_urgency.state import UrgencyAgentState

__all__ = [
    "TurnUrgencyConfig",
    "TurnUrgencyMiddleware",
    "UrgencyAgentState",
    "UrgencyCalculator",
    "UrgencyLevel",
    "UrgencyPrompts",
    "UrgencyState",
    "UrgencyThresholds",
]
