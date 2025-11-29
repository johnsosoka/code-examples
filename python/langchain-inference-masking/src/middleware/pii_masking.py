"""
PII Masking Middleware for LangChain agents.

This middleware intercepts messages before LLM inference, replaces detected PII
with hash placeholders, and restores original values in responses after inference.
"""

import hashlib
import logging
import re
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

# PII detection patterns (basic set: email, phone, SSN)
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"(?:\+1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def _generate_placeholder(value: str, pii_type: str) -> str:
    """Generate a deterministic placeholder for a PII value."""
    hash_digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"[{pii_type.upper()}:{hash_digest}]"


class PiiMaskingMiddleware(AgentMiddleware):
    """
    Middleware that masks PII in messages before LLM inference,
    then restores original values in responses.

    Supported PII types:
        - Email addresses
        - Phone numbers (US format)
        - Social Security Numbers (SSN)

    The middleware maintains an in-memory registry mapping placeholders
    to original values, enabling round-trip masking/unmasking.

    Example usage:
        from langchain.agents import create_agent

        middleware = PiiMaskingMiddleware()
        agent = create_agent(
            model="gpt-4o",
            tools=[],
            middleware=[middleware]
        )
    """

    def __init__(self) -> None:
        super().__init__()
        self._mask_registry: dict[str, str] = {}  # placeholder -> original value

    def _mask_pii_in_text(self, text: str) -> str:
        """Detect and replace PII in text with placeholders."""
        masked_text = text

        for pii_type, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(text):
                original_value = match.group()
                placeholder = _generate_placeholder(original_value, pii_type)

                # Store mapping for later restoration
                self._mask_registry[placeholder] = original_value

                masked_text = masked_text.replace(original_value, placeholder)
                logger.debug(f"Masked {pii_type}: {original_value} -> {placeholder}")

        return masked_text

    def _unmask_pii_in_text(self, text: str) -> str:
        """Restore original PII values from placeholders."""
        unmasked_text = text

        for placeholder, original_value in self._mask_registry.items():
            if placeholder in unmasked_text:
                unmasked_text = unmasked_text.replace(placeholder, original_value)
                logger.debug(f"Unmasked: {placeholder} -> {original_value}")

        return unmasked_text

    def _mask_message(self, message: BaseMessage) -> BaseMessage:
        """Create a new message with masked content."""
        if not isinstance(message.content, str):
            return message

        masked_content = self._mask_pii_in_text(message.content)

        if masked_content == message.content:
            return message

        # Create new message of same type with masked content
        if isinstance(message, HumanMessage):
            return HumanMessage(content=masked_content)
        elif isinstance(message, AIMessage):
            return AIMessage(content=masked_content)
        else:
            # For other message types, return as-is
            return message

    def _unmask_message(self, message: BaseMessage) -> BaseMessage:
        """Create a new message with unmasked content."""
        if not isinstance(message.content, str):
            return message

        unmasked_content = self._unmask_pii_in_text(message.content)

        if unmasked_content == message.content:
            return message

        if isinstance(message, AIMessage):
            return AIMessage(content=unmasked_content)
        elif isinstance(message, HumanMessage):
            return HumanMessage(content=unmasked_content)
        else:
            return message

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        Wrap the model call to mask PII before and unmask after.

        This hook intercepts the actual model request, masks PII in messages,
        calls the model, then restores PII in the response.

        Args:
            request: The model request containing messages to be sent
            handler: The function to call the model

        Returns:
            ModelResponse with PII restored in the output
        """
        # Mask PII in all messages before sending to LLM
        masked_messages: list[AnyMessage] = []
        for msg in request.messages:
            masked_messages.append(self._mask_message(msg))  # type: ignore

        pii_count = len(self._mask_registry)
        if pii_count > 0:
            logger.info(f"Masked {pii_count} PII value(s) before model call")

        # Create new request with masked messages
        masked_request = request.override(messages=masked_messages)

        # Call the model with masked messages
        response = handler(masked_request)

        # Unmask PII in the response messages
        if response.result:
            unmasked_results: list[BaseMessage] = []
            modified = False

            for msg in response.result:
                if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                    unmasked_content = self._unmask_pii_in_text(msg.content)
                    if unmasked_content != msg.content:
                        unmasked_results.append(AIMessage(content=unmasked_content))
                        modified = True
                    else:
                        unmasked_results.append(msg)
                else:
                    unmasked_results.append(msg)

            if modified:
                logger.info("Restored PII in model response")
                return ModelResponse(
                    result=unmasked_results,
                    structured_response=response.structured_response,
                )

        return response

    def clear_registry(self) -> None:
        """Clear the PII mask registry. Call between conversations if needed."""
        self._mask_registry.clear()
        logger.debug("Cleared PII mask registry")
