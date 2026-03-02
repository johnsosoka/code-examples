"""Shared Pydantic models for structured outputs."""

from typing import List, Optional
from pydantic import BaseModel, Field


class SelectionResult(BaseModel):
    """Result when LLM selects from indexed options."""
    selected_indices: List[int] = Field(
        description="The indices selected from the provided options"
    )
    reasoning: str = Field(
        description="Brief explanation of why these indices were selected"
    )


class QuoteSelection(BaseModel):
    """Structured output for quote selection."""
    selected_indices: List[int] = Field(
        description="Indices of sentences that should be quoted"
    )
    explanation: str = Field(
        description="Why these specific sentences were selected"
    )


class ElementSelection(BaseModel):
    """Structured output for web element selection."""
    selected_index: int = Field(
        description="The index of the element to interact with"
    )
    action: str = Field(
        description="The action to perform (click, type, etc.)"
    )
    reasoning: str = Field(
        description="Why this element and action were selected"
    )


class QuerySelection(BaseModel):
    """Structured output for SQL query template selection."""
    template_index: int = Field(
        description="The index of the query template to use"
    )
    parameters: dict[str, str] = Field(
        description="Parameters to fill into the template"
    )
    reasoning: str = Field(
        description="Why this template was selected"
    )


class ConfigSelection(BaseModel):
    """Structured output for configuration selection."""
    config_index: int = Field(
        description="The index of the configuration to use"
    )
    reasoning: str = Field(
        description="Why this configuration was selected for the task"
    )
