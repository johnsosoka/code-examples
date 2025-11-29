"""Global singleton registry for PII placeholder-to-value mappings."""

import logging

logger = logging.getLogger(__name__)


class PiiRegistry:
    """
    Global singleton for PII placeholder-to-value mappings.

    Enables tools to look up real PII values from masked placeholders
    that were registered by the middleware.
    """

    _instance: "PiiRegistry | None" = None
    _registry: dict[str, str]  # placeholder -> original

    def __new__(cls) -> "PiiRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PiiRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, placeholder: str, original: str) -> None:
        """Register a placeholder-to-original mapping."""
        self._registry[placeholder] = original
        logger.debug(f"Registered: {placeholder}")

    def lookup(self, placeholder: str) -> tuple[str | None, str | None]:
        """
        Look up the original value for a placeholder.

        Supports both exact match (e.g., '[PHONE:59c0b4a6]') and
        partial hash match (e.g., '59c0b4a6') for LLM flexibility.

        Returns:
            Tuple of (original_value, matched_placeholder) or (None, None)
        """
        # Exact match first
        if placeholder in self._registry:
            return self._registry[placeholder], placeholder

        # Try partial hash match (LLM may strip brackets)
        for key, value in self._registry.items():
            if placeholder in key:
                logger.debug(f"Partial match: {placeholder} -> {key}")
                return value, key

        return None, None

    def clear(self) -> None:
        """Clear all registered mappings."""
        self._registry.clear()
        logger.debug("Registry cleared")

    @property
    def registry(self) -> dict[str, str]:
        """Read-only access to the registry for debugging."""
        return dict(self._registry)
