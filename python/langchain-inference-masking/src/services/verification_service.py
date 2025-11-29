"""Mock verification service that requires REAL (unmasked) PII values."""

import logging

logger = logging.getLogger(__name__)


class VerificationService:
    """
    Mock verification service - simulates a backend that requires real PII.

    In a real system, this would call an external API or database
    that needs the actual phone number or SSN to perform verification.
    """

    def verify_phone(self, phone: str) -> dict:
        """
        Verify a phone number.

        Args:
            phone: The REAL phone number (not masked)

        Returns:
            Verification result dict
        """
        logger.info(f"VerificationService.verify_phone called with: {phone}")
        # Mock verification - in reality would call external service
        # Note: Do NOT return the real phone number - keep PII out of tool responses
        return {
            "status": "verified",
            "carrier": "Mock Carrier Inc.",
            "line_type": "mobile",
        }

    def verify_ssn(self, ssn: str) -> dict:
        """
        Verify a Social Security Number.

        Args:
            ssn: The REAL SSN (not masked)

        Returns:
            Verification result dict
        """
        logger.info(f"VerificationService.verify_ssn called with: {ssn}")
        # Mock verification - do NOT return any part of the SSN
        return {
            "status": "verified",
            "issuing_state": "Mock State",
        }
