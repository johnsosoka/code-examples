"""
Text-to-SQL Example: Free-Form vs Template Selection

This module demonstrates the security and reliability benefits of using
indexed query template selection over free-form SQL generation.

Examples:
    >>> from src.text_to_sql import wrong_way, right_way
    >>> wrong_way.demonstrate_risks()
    >>> right_way.demonstrate_safety()
"""

from .schema import USERS_TABLE, ORDERS_TABLE, QUERY_TEMPLATES
from .wrong_way import FreeFormSQLGenerator, demonstrate_risks
from .right_way import TemplateSQLGenerator, demonstrate_safety

__all__ = [
    "USERS_TABLE",
    "ORDERS_TABLE", 
    "QUERY_TEMPLATES",
    "FreeFormSQLGenerator",
    "TemplateSQLGenerator",
    "demonstrate_risks",
    "demonstrate_safety",
]
