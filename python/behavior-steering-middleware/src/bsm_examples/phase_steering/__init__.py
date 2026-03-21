"""Phase-Based Steering middleware for LangGraph agents.

This package provides a middleware implementation for guiding agent behavior
through distinct workflow phases (GATHER, ANALYZE, RECOMMEND, COMPLETE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PhaseBehavior, PhaseConfig, PhaseTransition
    from .detector import DefaultPhaseDetector, PhaseDetector
    from .middleware import PhaseSteeringMiddleware
    from .phase import Phase
    from .state import PhaseState

# Deferred imports to avoid circular dependencies
def __getattr__(name: str) -> type:
    if name == "Phase":
        from .phase import Phase

        return Phase
    elif name == "PhaseBehavior":
        from .config import PhaseBehavior

        return PhaseBehavior
    elif name == "PhaseConfig":
        from .config import PhaseConfig

        return PhaseConfig
    elif name == "PhaseTransition":
        from .config import PhaseTransition

        return PhaseTransition
    elif name == "PhaseState":
        from .state import PhaseState

        return PhaseState
    elif name == "PhaseDetector":
        from .detector import PhaseDetector

        return PhaseDetector
    elif name == "DefaultPhaseDetector":
        from .detector import DefaultPhaseDetector

        return DefaultPhaseDetector
    elif name == "PhaseSteeringMiddleware":
        from .middleware import PhaseSteeringMiddleware

        return PhaseSteeringMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Phase",
    "PhaseBehavior",
    "PhaseConfig",
    "PhaseDetector",
    "DefaultPhaseDetector",
    "PhaseState",
    "PhaseSteeringMiddleware",
    "PhaseTransition",
]
