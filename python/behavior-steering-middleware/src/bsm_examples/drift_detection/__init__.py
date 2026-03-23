"""Worker Check-In Monitor middleware for LangGraph agents.

Detects when an agent stops calling a designated tool (e.g. ``send_message``)
and injects a ``<framework_instruction>``-tagged reminder to nudge it back on track.
"""

from __future__ import annotations

from bsm_examples.drift_detection.detector import CheckInDetector
from bsm_examples.drift_detection.middleware import CheckInMiddleware
from bsm_examples.drift_detection.models import CheckInConfig
from bsm_examples.drift_detection.state import CheckInAgentState

__all__ = [
    "CheckInConfig",
    "CheckInAgentState",
    "CheckInDetector",
    "CheckInMiddleware",
]
