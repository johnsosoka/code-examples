"""Worker Check-In Monitor middleware for LangGraph agents.

Intercepts model calls to detect when the agent has gone too many turns
without calling the configured target tool (e.g. ``send_message``).
When the silence threshold is reached, a ``<framework_instruction>`` reminder
is injected into the last user or tool message before the model sees it.
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langgraph.types import Command

from bsm_examples.drift_detection.detector import CheckInDetector
from bsm_examples.drift_detection.models import CheckInConfig
from bsm_examples.drift_detection.state import CheckInAgentState
from bsm_examples.injection import inject_framework_instruction

logger = logging.getLogger(__name__)


class CheckInMiddleware(AgentMiddleware):
    """Middleware that monitors a specific tool and reminds the agent to use it.

    Steering content is injected as a ``<framework_instruction>`` block
    prepended to the last user or tool message — the system prompt is not
    modified and no new messages are appended. The agent's system prompt
    should mention that ``<framework_instruction>`` tags will appear with
    guidance.

    Example:
        >>> config = CheckInConfig(target_tool="send_message", max_silent_turns=2)
        >>> middleware = CheckInMiddleware(config)
        >>> agent = create_agent(
        ...     model=ChatOpenAI(model="gpt-4.1-mini"),
        ...     tools=[send_message, do_research],
        ...     middleware=[middleware],
        ...     state_schema=CheckInAgentState,
        ...     system_prompt=(
        ...         "You are a research agent. Send regular status updates. "
        ...         "You will receive <framework_instruction> tags with "
        ...         "guidance — follow them."
        ...     ),
        ... )
    """

    state_schema = CheckInAgentState

    def __init__(self, config: CheckInConfig) -> None:
        """Initialise the middleware.

        Args:
            config: Check-in monitoring configuration.
        """
        super().__init__()
        self._config = config
        self._detector = CheckInDetector(config)

    @property
    def detector(self) -> CheckInDetector:
        """Expose the underlying detector for observability and testing."""
        return self._detector

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Intercept the model call to inject check-in reminders when needed.

        Steps:
        1. If the detector says to intervene, inject a framework_instruction
           into the last user/tool message via ``request.override``.
        2. Call the handler (the real model).
        3. Record the turn so counters stay accurate.
        4. Return an ExtendedModelResponse that writes check-in fields back
           into the agent state.

        Args:
            request: Incoming model request with messages and metadata.
            handler: Callable that executes the actual LLM call.

        Returns:
            ExtendedModelResponse with updated check-in state fields.
        """
        if self._detector.should_intervene():
            reminder = self._detector.build_reminder()
            modified_messages = inject_framework_instruction(
                list(request.messages), reminder,
            )
            request = request.override(messages=modified_messages)
            self._detector.record_intervention()
            logger.warning(
                "Check-in intervention: agent silent for %d turns without %s",
                self._detector.turns_since_target,
                self._config.target_tool,
            )

        response = handler(request)

        if response.result:
            self._detector.record_turn(response.result)

        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "turns_since_check_in": self._detector.turns_since_target,
                    "intervention_count": self._detector.intervention_count,
                }
            ),
        )

    def reset(self) -> None:
        """Reset the middleware for a new conversation."""
        self._detector = CheckInDetector(self._config)
        logger.debug("CheckInMiddleware reset")
