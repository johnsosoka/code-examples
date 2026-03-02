"""
Example 4: Configuration Selection - Free-Form vs Validated Templates

This example demonstrates the difference between:
- WRONG WAY: LLM generates configuration free-form (error-prone)
- RIGHT WAY: LLM selects from pre-validated config templates (guaranteed valid)

The key insight: LLMs are great at reasoning about which config to use,
but terrible at generating syntactically and semantically valid configurations.
"""

from .config_models import LLMConfig, DatabaseConfig
from .wrong_way import WrongWayConfigSelector
from .right_way import RightWayConfigSelector

__all__ = [
    "LLMConfig",
    "DatabaseConfig", 
    "WrongWayConfigSelector",
    "RightWayConfigSelector",
]