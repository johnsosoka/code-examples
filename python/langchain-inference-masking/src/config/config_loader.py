"""Configuration loader for the PII masking example."""

import os
from pathlib import Path

import yaml


class ConfigLoader:
    """Loads configuration from YAML file with environment variable fallbacks."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "config.yml"
        self._config_path = Path(config_path)
        self._config: dict = {}
        self._load()

    def _load(self) -> None:
        if self._config_path.exists():
            with open(self._config_path) as f:
                self._config = yaml.safe_load(f) or {}

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from config or OPENAI_API_KEY env var."""
        return (
            self._config.get("api_keys", {}).get("openai")
            or os.environ.get("OPENAI_API_KEY", "")
        )

    @property
    def model_name(self) -> str:
        return self._config.get("model", {}).get("name", "gpt-4o-mini")
