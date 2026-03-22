"""Turn-Based Urgency middleware for LangGraph agents.

This package provides middleware that dynamically adjusts agent behavior
based on remaining turn budget, creating progressive behavior escalation
from exploratory to urgent as turns are consumed.

Example:
    >>> from bsm_examples.turn_urgency import TurnUrgencyConfig, TurnUrgencyMiddleware
    >>> config = TurnUrgencyConfig(max_turns=5)
    >>> middleware = TurnUrgencyMiddleware(config)
"""

from __future__ import annotations

from bsm_examples.turn_urgency.middleware import (
    TurnUrgencyMiddleware,
    UrgencyCalculator,
    UrgencyState,
)
from bsm_examples.turn_urgency.types import (
    TurnUrgencyConfig,
    UrgencyAgentState,
    UrgencyLevel,
    UrgencyPrompts,
    UrgencyThresholds,
)

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
