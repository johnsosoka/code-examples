"""Phase enumeration for workflow phase definitions."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self


class Phase(Enum):
    """Enumeration of workflow phases for the research assistant demo.

    Phases represent distinct stages in a multi-step workflow, each with
    specific objectives, allowed tools, and behavioral constraints.

    Attributes:
        GATHER: Information collection phase - focus on acquiring raw data.
        ANALYZE: Pattern identification and evaluation phase - process data.
        RECOMMEND: Synthesis and recommendation phase - formulate conclusions.
        COMPLETE: Final state, workflow concluded - present results.

    Example:
        >>> phase = Phase.GATHER
        >>> phase.display_name
        'Gather'
        >>> phase.order
        0
    """

    GATHER = auto()  # Information collection phase
    ANALYZE = auto()  # Pattern identification and evaluation phase
    RECOMMEND = auto()  # Synthesis and recommendation phase
    COMPLETE = auto()  # Final state, workflow concluded

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Parse phase from string representation.

        Args:
            value: String name of the phase (case-insensitive).

        Returns:
            The corresponding Phase enum member.

        Raises:
            ValueError: If the string does not match any phase.

        Example:
            >>> Phase.from_string("gather")
            <Phase.GATHER: 1>
            >>> Phase.from_string("ANALYZE")
            <Phase.ANALYZE: 2>
        """
        try:
            return cls[value.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown phase: {value}") from e

    @property
    def display_name(self) -> str:
        """Human-readable phase name.

        Returns:
            Title-cased phase name.

        Example:
            >>> Phase.GATHER.display_name
            'Gather'
        """
        return self.name.title()

    @property
    def order(self) -> int:
        """Numeric order for phase progression validation.

        Returns:
            Zero-based index of the phase in the sequence.

        Example:
            >>> Phase.GATHER.order
            0
            >>> Phase.ANALYZE.order
            1
        """
        return list(Phase).index(self)
