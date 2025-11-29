"""Validation script to test middleware integration with PiiRegistry."""

import logging

from middleware.pii_masking import PiiMaskingMiddleware
from services.pii_registry import PiiRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_middleware_registry_integration():
    """Test that middleware properly uses the PiiRegistry singleton."""
    logger.info("Testing middleware integration with PiiRegistry...")

    # Clear registry to start fresh
    registry = PiiRegistry.get_instance()
    registry.clear()

    # Create middleware instance
    middleware = PiiMaskingMiddleware()

    # Test that middleware uses the same singleton instance
    test_text = "My phone is 555-867-5309 and SSN is 123-45-6789"
    masked_text = middleware._mask_pii_in_text(test_text)

    logger.info(f"Original: {test_text}")
    logger.info(f"Masked: {masked_text}")
    logger.info(f"Registry: {registry.registry}")

    # Verify placeholders were created
    assert "[PHONE:" in masked_text
    assert "[SSN:" in masked_text
    assert "555-867-5309" not in masked_text
    assert "123-45-6789" not in masked_text

    # Verify registry was populated
    assert len(registry.registry) == 2
    logger.info(f"Registry has {len(registry.registry)} entries")

    # Test unmasking
    unmasked_text = middleware._unmask_pii_in_text(masked_text)
    logger.info(f"Unmasked: {unmasked_text}")

    assert unmasked_text == test_text
    logger.info("Middleware properly integrates with PiiRegistry!")

    # Test backwards compatibility property
    mask_registry_dict = middleware._mask_registry
    logger.info(f"Backwards compatible _mask_registry property: {mask_registry_dict}")
    assert isinstance(mask_registry_dict, dict)
    assert len(mask_registry_dict) == 2

    # Test clear_registry method
    middleware.clear_registry()
    assert len(registry.registry) == 0
    logger.info("clear_registry() method works correctly")

    logger.info("All middleware integration tests passed!")


def main():
    """Run middleware validation tests."""
    logger.info("=" * 60)
    logger.info("Starting middleware validation tests...")
    logger.info("=" * 60)

    try:
        test_middleware_registry_integration()

        logger.info("=" * 60)
        logger.info("All middleware validation tests passed!")
        logger.info("=" * 60)

    except AssertionError as e:
        logger.error(f"Validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
