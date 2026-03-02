"""
Pydantic models for configuration validation.

These models enforce strict validation rules that configurations must satisfy.
Any configuration that doesn't meet these constraints will fail validation.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """
    Configuration for LLM API calls.
    
    Enforces:
    - Only approved models can be used
    - Temperature within valid range (0.0 - 2.0)
    - Token limits within bounds
    - Timeout and retry constraints
    """
    
    model: str = Field(
        ...,
        description="LLM model identifier",
        examples=["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
    )
    temperature: float = Field(
        ...,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = very random)"
    )
    max_tokens: int = Field(
        ...,
        ge=1,
        le=4096,
        description="Maximum tokens to generate"
    )
    timeout: int = Field(
        ...,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )
    retries: int = Field(
        ...,
        ge=0,
        le=5,
        description="Number of retry attempts on failure"
    )
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Ensure only approved models are used."""
        allowed_models = {"gpt-4", "gpt-3.5-turbo", "claude-3-opus"}
        if v not in allowed_models:
            raise ValueError(
                f"Invalid model '{v}'. Must be one of: {allowed_models}"
            )
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 1000,
                "timeout": 30,
                "retries": 3
            }
        }


class DatabaseConfig(BaseModel):
    """
    Configuration for database connections.
    
    Enforces:
    - Valid port range (1-65535)
    - Reasonable connection pool sizes
    """
    
    host: str = Field(
        ...,
        description="Database host address",
        examples=["localhost", "db.example.com"]
    )
    port: int = Field(
        ...,
        ge=1,
        le=65535,
        description="Database port number"
    )
    pool_size: int = Field(
        ...,
        ge=1,
        le=100,
        description="Connection pool size"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "host": "localhost",
                "port": 5432,
                "pool_size": 10
            }
        }


# Pre-validated configuration templates
# These are GUARANTEED to be valid - they were validated at import time
CONFIG_TEMPLATES = {
    0: {
        "name": "production_llm",
        "description": "Conservative settings for production use",
        "config": LLMConfig(
            model="gpt-4",
            temperature=0.0,
            max_tokens=1000,
            timeout=30,
            retries=3
        )
    },
    1: {
        "name": "creative_llm",
        "description": "Higher temperature for creative tasks",
        "config": LLMConfig(
            model="gpt-4",
            temperature=0.8,
            max_tokens=2000,
            timeout=60,
            retries=2
        )
    },
    2: {
        "name": "fast_llm",
        "description": "Quick responses with cheaper model",
        "config": LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=500,
            timeout=10,
            retries=1
        )
    },
    3: {
        "name": "reliable_llm",
        "description": "Maximum reliability for critical tasks",
        "config": LLMConfig(
            model="gpt-4",
            temperature=0.0,
            max_tokens=500,
            timeout=60,
            retries=5
        )
    },
}


def get_template_descriptions() -> dict[int, str]:
    """
    Get human-readable descriptions of all config templates.
    
    Returns:
        Dictionary mapping index to formatted description string
    """
    return {
        idx: f"[{idx}] {template['name']}: {template['description']}"
        for idx, template in CONFIG_TEMPLATES.items()
    }


def get_template_by_index(index: int) -> dict | None:
    """
    Retrieve a configuration template by its index.
    
    Args:
        index: Template index
        
    Returns:
        Template dictionary or None if index not found
    """
    return CONFIG_TEMPLATES.get(index)