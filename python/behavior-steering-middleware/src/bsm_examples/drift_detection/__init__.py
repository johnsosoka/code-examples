"""Worker Heartbeat Monitor middleware for LangGraph agents.

Detects when an agent stops calling a designated tool (e.g. ``send_message``)
and injects a ``<system>``-tagged reminder to nudge it back on track.
"""

from __future__ import annotations

from bsm_examples.drift_detection.detector import HeartbeatDetector
from bsm_examples.drift_detection.middleware import HeartbeatMiddleware
from bsm_examples.drift_detection.models import HeartbeatConfig
from bsm_examples.drift_detection.state import HeartbeatAgentState

__all__ = [
    "HeartbeatConfig",
    "HeartbeatAgentState",
    "HeartbeatDetector",
    "HeartbeatMiddleware",
]
