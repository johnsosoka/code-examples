"""Shared utility for injecting framework instructions into agent messages.

All behavior-steering middlewares use ``inject_framework_instruction`` to
prepend a ``<framework_instruction>``-tagged block to the last user or tool
message in the conversation. This keeps the API call structure valid (no
rogue SystemMessages at the end of the list) and places the instruction
where the model will see it immediately before generating a response.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)


def inject_framework_instruction(
    messages: list[BaseMessage],
    instruction: str,
) -> list[AnyMessage]:
    """Prepend a framework instruction to the last user or tool message.

    The instruction is wrapped in ``<framework_instruction>`` tags and placed
    **before** the original message content so the model reads the directive
    first.

    If no suitable message is found the list is returned unmodified (this
    should not happen in normal agent operation).

    Args:
        messages: Conversation messages from ``request.messages``.
        instruction: Raw guidance text (tags are added automatically).

    Returns:
        A shallow copy of the message list with the targeted message replaced.
    """
    modified: list[AnyMessage] = list(messages)  # type: ignore[arg-type]
    tagged = f"<framework_instruction>\n{instruction}\n</framework_instruction>"

    for i in range(len(modified) - 1, -1, -1):
        msg = modified[i]
        if isinstance(msg, HumanMessage):
            modified[i] = HumanMessage(
                content=f"{tagged}\n\n{msg.content}",
            )
            logger.debug("Injected framework_instruction into HumanMessage at index %d", i)
            return modified
        if isinstance(msg, ToolMessage):
            modified[i] = ToolMessage(
                content=f"{tagged}\n\n{msg.content}",
                tool_call_id=msg.tool_call_id,
            )
            logger.debug("Injected framework_instruction into ToolMessage at index %d", i)
            return modified

    logger.warning("No HumanMessage or ToolMessage found — instruction not injected")
    return modified
