"""LangChain tools for identity verification with PII masking support."""

import logging

from langchain_core.tools import tool

from services.pii_registry import PiiRegistry
from services.verification_service import VerificationService

logger = logging.getLogger(__name__)


@tool
def verify_identity(identifier: str) -> str:
    """
    Verify a phone number or SSN. Pass the identifier exactly as provided.

    Use this tool when the user asks to verify their phone number or SSN.
    Pass the identifier exactly as it appears in the conversation.

    Args:
        identifier: The phone number or SSN to verify (pass exactly as given)

    Returns:
        Verification result message
    """
    logger.info(f"verify_identity tool received: {identifier}")

    # Look up the real value from the global registry
    registry = PiiRegistry.get_instance()
    real_value, matched_placeholder = registry.lookup(identifier)

    if real_value is None:
        logger.warning(f"Identifier not found in registry: {identifier}")
        return "Unable to verify: identifier not found in system"

    logger.info(f"Resolved {identifier} -> {real_value} (matched: {matched_placeholder})")

    # Call the verification service with the REAL value
    service = VerificationService()

    # Use matched_placeholder to determine type (it has the [TYPE:hash] format)
    if matched_placeholder and "PHONE" in matched_placeholder.upper():
        result = service.verify_phone(real_value)
        return f"Phone verification complete: {result}"
    elif matched_placeholder and "SSN" in matched_placeholder.upper():
        result = service.verify_ssn(real_value)
        return f"SSN verification complete: {result}"
    else:
        logger.warning(f"Unknown identifier type: {matched_placeholder}")
        return "Unable to determine identifier type for verification"
