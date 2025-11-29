"""Validation script to test PII registry and tool integration."""

import logging

from services.pii_registry import PiiRegistry
from services.verification_service import VerificationService
from tools.verification_tools import verify_identity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_pii_registry():
    """Test the PiiRegistry singleton."""
    logger.info("Testing PiiRegistry singleton...")

    # Get instance and register some PII
    registry = PiiRegistry.get_instance()
    registry.clear()  # Start fresh

    registry.register("[PHONE:59c0b4a6]", "555-867-5309")
    registry.register("[SSN:abc12345]", "123-45-6789")

    # Test lookup
    phone = registry.lookup("[PHONE:59c0b4a6]")
    ssn = registry.lookup("[SSN:abc12345]")

    assert phone == "555-867-5309", f"Expected 555-867-5309, got {phone}"
    assert ssn == "123-45-6789", f"Expected 123-45-6789, got {ssn}"

    logger.info("PiiRegistry tests passed!")
    logger.info(f"Registry contents: {registry.registry}")


def test_verification_service():
    """Test the VerificationService."""
    logger.info("Testing VerificationService...")

    service = VerificationService()

    # Test phone verification
    phone_result = service.verify_phone("555-867-5309")
    assert phone_result["verified"] is True
    assert phone_result["phone"] == "555-867-5309"
    logger.info(f"Phone verification result: {phone_result}")

    # Test SSN verification
    ssn_result = service.verify_ssn("123-45-6789")
    assert ssn_result["verified"] is True
    assert ssn_result["ssn_last_four"] == "6789"
    logger.info(f"SSN verification result: {ssn_result}")

    logger.info("VerificationService tests passed!")


def test_verification_tool():
    """Test the verify_identity tool."""
    logger.info("Testing verify_identity tool...")

    # Set up registry with test data
    registry = PiiRegistry.get_instance()
    registry.clear()
    registry.register("[PHONE:59c0b4a6]", "555-867-5309")
    registry.register("[SSN:abc12345]", "123-45-6789")

    # Test phone verification through tool
    phone_result = verify_identity.invoke("[PHONE:59c0b4a6]")
    logger.info(f"Phone tool result: {phone_result}")
    assert "verification complete" in phone_result.lower()

    # Test SSN verification through tool
    ssn_result = verify_identity.invoke("[SSN:abc12345]")
    logger.info(f"SSN tool result: {ssn_result}")
    assert "verification complete" in ssn_result.lower()

    # Test unknown identifier
    unknown_result = verify_identity.invoke("[UNKNOWN:test]")
    logger.info(f"Unknown tool result: {unknown_result}")
    assert "not found" in unknown_result.lower()

    logger.info("verify_identity tool tests passed!")


def main():
    """Run all validation tests."""
    logger.info("=" * 60)
    logger.info("Starting validation tests...")
    logger.info("=" * 60)

    try:
        test_pii_registry()
        logger.info("")

        test_verification_service()
        logger.info("")

        test_verification_tool()
        logger.info("")

        logger.info("=" * 60)
        logger.info("All validation tests passed!")
        logger.info("=" * 60)

    except AssertionError as e:
        logger.error(f"Validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
