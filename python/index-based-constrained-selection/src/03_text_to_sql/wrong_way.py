"""
WRONG WAY: Free-form SQL generation by LLM.

This module demonstrates the dangers of allowing LLMs to generate SQL directly:
- SQL injection vulnerabilities
- Syntax errors
- Incorrect table/column references
- Potential for destructive operations

⚠️ WARNING: This approach is shown for educational purposes only.
DO NOT use free-form SQL generation in production systems.
"""

import re
from typing import Any
from dataclasses import dataclass

from .schema import get_schema_description, USERS_TABLE, ORDERS_TABLE


@dataclass
class SQLGenerationResult:
    """Result of SQL generation attempt."""
    query: str
    is_valid: bool
    risks: list[str]
    warnings: list[str]


class FreeFormSQLGenerator:
    """
    Simulates an LLM generating SQL queries free-form.
    
    This class demonstrates what happens when you trust an LLM to write SQL
    without constraints. Spoiler: it's not pretty.
    
    Example:
        >>> generator = FreeFormSQLGenerator()  # doctest: +SKIP
        >>> result = generator.generate("Find orders for user 123")  # doctest: +SKIP
        >>> print(result.query)  # doctest: +SKIP
        SELECT * FROM orders WHERE user_id = 123
        >>> print(result.risks)  # doctest: +SKIP
        ['Potential SQL injection: unparameterized user input']
    """
    
    # Dangerous SQL keywords that should never appear in generated queries
    DANGEROUS_KEYWORDS = [
        "DROP", "DELETE", "TRUNCATE", "ALTER", "GRANT", "REVOKE",
        "EXEC", "EXECUTE", "UNION", "INTO OUTFILE", "LOAD_FILE"
    ]
    
    def __init__(self):
        """Initialize the free-form SQL generator."""
        self.schema_description = get_schema_description()
        self.generation_count = 0
    
    def generate(self, user_request: str) -> SQLGenerationResult:
        """
        Simulate LLM generating SQL from natural language request.
        
        In a real implementation, this would call an LLM API. Here we simulate
        various problematic outputs to demonstrate the risks.
        
        Args:
            user_request: Natural language description of desired query
            
        Returns:
            SQLGenerationResult containing the generated query and risk analysis
        """
        self.generation_count += 1
        
        # Simulate different problematic LLM outputs based on request patterns
        query, risks, warnings = self._simulate_llm_output(user_request)
        
        # Validate the generated query
        is_valid = self._validate_syntax(query)
        risks.extend(self._detect_risks(query))
        
        return SQLGenerationResult(
            query=query,
            is_valid=is_valid,
            risks=risks,
            warnings=warnings
        )
    
    def _simulate_llm_output(self, request: str) -> tuple[str, list[str], list[str]]:
        """
        Simulate various problematic LLM outputs.
        
        This demonstrates common failure modes of free-form SQL generation.
        """
        request_lower = request.lower()
        risks: list[str] = []
        warnings: list[str] = []
        
        # Simulate SQL injection vulnerability
        if "user" in request_lower and ("123" in request or "id" in request_lower):
            # LLM might embed user input directly without parameterization
            if "find" in request_lower or "get" in request_lower:
                query = "SELECT * FROM orders WHERE user_id = 123"
                risks.append("SQL injection risk: User ID hardcoded in query")
                return query, risks, warnings
        
        # Simulate wrong table name hallucination
        if "order" in request_lower:
            if self.generation_count % 3 == 0:  # Every 3rd call fails
                query = "SELECT * FROM order WHERE user_id = 123"  # Wrong: 'order' not 'orders'
                risks.append("Invalid table name: 'order' does not exist (should be 'orders')")
                return query, risks, warnings
        
        # Simulate SQL injection via malicious input
        if "'" in request or ";" in request or "--" in request:
            # Simulate LLM failing to sanitize input
            query = f"SELECT * FROM users WHERE name = '{request.split('name')[-1].strip()}'"
            risks.append("CRITICAL: SQL injection vulnerability - unsanitized user input")
            risks.append("Malicious input could execute arbitrary SQL commands")
            return query, risks, warnings
        
        # Simulate destructive operation generation
        if "delete" in request_lower or "remove" in request_lower or "drop" in request_lower:
            query = "DROP TABLE users"  # OH NO
            risks.append("CRITICAL: Destructive operation generated!")
            risks.append("This would delete the entire users table!")
            return query, risks, warnings
        
        # Simulate syntax error
        if "count" in request_lower:
            if self.generation_count % 2 == 0:
                query = "SELECT COUNT(*) FROM orders WHERE user_id = 123 GROUP"  # Incomplete
                risks.append("Syntax error: Incomplete GROUP BY clause")
                return query, risks, warnings
        
        # Simulate correct query (rare case)
        query = "SELECT * FROM orders WHERE user_id = 123 ORDER BY created_at DESC"
        warnings.append("Even 'correct' queries may have subtle injection risks")
        return query, risks, warnings
    
    def _validate_syntax(self, query: str) -> bool:
        """
        Basic syntax validation (simplified).
        
        In reality, you'd need a full SQL parser. This is just for demonstration.
        """
        # Check for basic structure
        if not query.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE")):
            if not any(kw in query.upper() for kw in ["DROP", "CREATE", "ALTER"]):
                return False
        
        # Check for unmatched quotes
        if query.count("'") % 2 != 0:
            return False
        
        # Check for incomplete clauses
        if query.upper().endswith(("WHERE", "GROUP", "ORDER", "FROM", "JOIN")):
            return False
        
        return True
    
    def _detect_risks(self, query: str) -> list[str]:
        """
        Detect security risks in generated SQL.
        
        Args:
            query: The SQL query to analyze
            
        Returns:
            List of detected risk descriptions
        """
        risks: list[str] = []
        query_upper = query.upper()
        
        # Check for dangerous keywords
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in query_upper:
                risks.append(f"Dangerous keyword detected: {keyword}")
        
        # Check for unparameterized values (simplified detection)
        # Look for numbers or strings that look like user input
        if re.search(r"=\s*\d+", query) and "?" not in query and ":" not in query:
            risks.append("Unparameterized numeric value detected")
        
        if re.search(r"=\s*'[^']+'", query) and "?" not in query:
            risks.append("Unparameterized string value detected (SQL injection risk)")
        
        # Check for SELECT * (performance anti-pattern)
        if "SELECT *" in query_upper:
            risks.append("Uses SELECT * (performance anti-pattern)")
        
        return risks
    
    def get_system_prompt(self) -> str:
        """
        Return the system prompt that would be used for free-form generation.
        
        This demonstrates how complex and error-prone the prompt engineering becomes.
        """
        return f"""You are a SQL expert. Generate SQL queries based on user requests.

Database Schema:
{self.schema_description}

Rules:
- Only generate SELECT statements (NO DELETE, DROP, ALTER, etc.)
- Use parameterized queries with ? placeholders
- Validate table and column names against the schema
- Never include user input directly in the query
- Use proper SQL syntax

Generate SQL for the following request:"""


def demonstrate_risks() -> None:
    """
    Demonstrate the various risks of free-form SQL generation.
    
    This function shows multiple failure modes that can occur when
    LLMs generate SQL without constraints.
    """
    generator = FreeFormSQLGenerator()
    
    print("=" * 70)
    print("WRONG WAY: Free-Form SQL Generation")
    print("=" * 70)
    print("\nDemonstrating various failure modes...\n")
    
    test_requests = [
        "Find all orders for user 123",
        "Find all orders for user 123",  # May hallucinate wrong table
        "Find all orders for user 123",  # May have syntax error
        "Get count of orders for user 456",
        "Get count of orders for user 456",  # May have syntax error
        "Find user with name 'Robert'); DROP TABLE users; --",
        "Delete all inactive users",
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n{'─' * 70}")
        print(f"Request {i}: {request}")
        print("─" * 70)
        
        result = generator.generate(request)
        
        print(f"Generated SQL:\n  {result.query}")
        print(f"\nSyntax Valid: {'✓ Yes' if result.is_valid else '✗ No'}")
        
        if result.risks:
            print(f"\n🚨 RISKS DETECTED:")
            for risk in result.risks:
                print(f"  • {risk}")
        
        if result.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        if not result.risks and not result.warnings:
            print("\n✓ No immediate risks detected (but still not safe!)")
    
    print(f"\n{'=' * 70}")
    print("SUMMARY: Free-form generation is inherently unsafe")
    print("=" * 70)
    print("""
Problems demonstrated:
1. SQL injection vulnerabilities from unsanitized input
2. Hallucinated table/column names
3. Syntax errors in generated queries
4. Potential for destructive operations (DROP, DELETE)
5. Inconsistent output format
6. Complex prompt engineering requirements

The LLM cannot be trusted to always follow instructions.
Even with careful prompting, edge cases will cause failures.
""")


if __name__ == "__main__":
    demonstrate_risks()
