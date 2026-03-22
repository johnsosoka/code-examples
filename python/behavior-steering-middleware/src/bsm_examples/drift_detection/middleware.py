"""Worker Heartbeat Monitor middleware for LangGraph agents.

Intercepts model calls to detect when the agent has gone too many turns
without calling the configured target tool (e.g. ``send_message``).
When the silence threshold is reached, a ``<system>``-tagged reminder is
appended to the message list before the model sees it.
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
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from bsm_examples.drift_detection.detector import HeartbeatDetector
from bsm_examples.drift_detection.models import HeartbeatConfig
from bsm_examples.drift_detection.state import HeartbeatAgentState

logger = logging.getLogger(__name__)


class HeartbeatMiddleware(AgentMiddleware):
    """Middleware that monitors a specific tool and reminds the agent to use it.

    Steering content is injected as a ``<system>``-tagged ``SystemMessage``
    appended to the conversation — the system prompt is not modified. The
    agent's system prompt should mention that ``<system>`` tags will appear
    with guidance so the model learns to act on them.

    Example:
        >>> config = HeartbeatConfig(target_tool="send_message", max_silent_turns=2)
        >>> middleware = HeartbeatMiddleware(config)
        >>> agent = create_agent(
        ...     model=ChatOpenAI(model="gpt-4.1-mini"),
        ...     tools=[send_message, do_research],
        ...     middleware=[middleware],
        ...     state_schema=HeartbeatAgentState,
        ...     system_prompt=(
        ...         "You are a research agent. Send regular status updates. "
        ...         "You will receive <system> tags with guidance — follow them."
        ...     ),
        ... )
    """

    state_schema = HeartbeatAgentState

    def __init__(self, config: HeartbeatConfig) -> None:
        """Initialise the middleware.

        Args:
            config: Heartbeat monitoring configuration.
        """
        super().__init__()
        self._config = config
        self._detector = HeartbeatDetector(config)

    @property
    def detector(self) -> HeartbeatDetector:
        """Expose the underlying detector for observability and testing."""
        return self._detector

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Intercept the model call to inject heartbeat reminders when needed.

        Steps:
        1. If the detector says to intervene, append a ``<system>`` reminder
           and override the request messages before calling the handler.
        2. Call the handler (the real model).
        3. Record the turn so counters stay accurate.
        4. Return an ExtendedModelResponse that writes heartbeat fields back
           into the agent state.

        Args:
            request: Incoming model request with messages and metadata.
            handler: Callable that executes the actual LLM call.

        Returns:
            ExtendedModelResponse with updated heartbeat state fields.
        """
        if self._detector.should_intervene():
            reminder = self._detector.build_reminder()
            steering_msg = SystemMessage(content=f"<system>{reminder}</system>")
            modified_messages = list(request.messages) + [steering_msg]
            request = request.override(messages=modified_messages)
            self._detector.record_intervention()
            logger.warning(
                "Heartbeat intervention: agent silent for %d turns without %s",
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
                    "turns_since_heartbeat": self._detector.turns_since_target,
                    "intervention_count": self._detector.intervention_count,
                }
            ),
        )

    def reset(self) -> None:
        """Reset the middleware for a new conversation."""
        self._detector = HeartbeatDetector(self._config)
        logger.debug("HeartbeatMiddleware reset")
