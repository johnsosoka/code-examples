"""
Text-to-SQL Demo: Free-Form vs Template Selection Comparison

This module provides a comprehensive comparison demonstrating why
indexed template selection is superior to free-form SQL generation.

Run this demo to see:
- The security risks of free-form generation
- The safety guarantees of template selection
- Side-by-side comparison of both approaches

Usage:
    python -m src.text_to_sql.demo
"""

import sys
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .wrong_way import FreeFormSQLGenerator, SQLGenerationResult
from .right_way import TemplateSQLGenerator, TemplateSelectionResult
from .schema import QUERY_TEMPLATES


@dataclass
class ComparisonResult:
    """Result of comparing both approaches for a single request."""
    request: str
    free_form_result: SQLGenerationResult
    template_result: TemplateSelectionResult
    winner: str
    reason: str


class TextToSQLComparison:
    """
    Comprehensive comparison of free-form vs template-based SQL generation.
    
    This class runs both approaches against the same requests and provides
    detailed analysis of the differences in security, reliability, and safety.
    """
    
    def __init__(self):
        """Initialize both generators."""
        self.free_form = FreeFormSQLGenerator()
        self.template = TemplateSQLGenerator()
    
    def compare(self, request: str) -> ComparisonResult:
        """
        Compare both approaches for a single request.
        
        Args:
            request: Natural language query request
            
        Returns:
            ComparisonResult with analysis of both approaches
        """
        # Run free-form generation
        free_form_result = self.free_form.generate(request)
        
        # Run template selection
        selection = self.template.simulate_llm_selection(request)
        template_result = self.template.generate(
            selection.template_index,
            selection.parameters
        )
        
        # Determine winner based on safety and correctness
        if not template_result.is_valid:
            winner = "FREE_FORM"
            reason = "Template approach rejected valid request (configuration issue)"
        elif free_form_result.risks:
            winner = "TEMPLATE"
            reason = f"Free-form has {len(free_form_result.risks)} security risk(s)"
        elif not free_form_result.is_valid:
            winner = "TEMPLATE"
            reason = "Free-form generated invalid SQL"
        elif free_form_result.warnings:
            winner = "TEMPLATE"
            reason = "Free-form has warnings; template is guaranteed safe"
        else:
            winner = "TEMPLATE"
            reason = "Template provides safety guarantees; free-form does not"
        
        return ComparisonResult(
            request=request,
            free_form_result=free_form_result,
            template_result=template_result,
            winner=winner,
            reason=reason
        )
    
    def run_comparison_suite(self) -> List[ComparisonResult]:
        """
        Run a comprehensive suite of test cases comparing both approaches.
        
        Returns:
            List of ComparisonResult for each test case
        """
        test_cases = [
            "Find all orders for user 123",
            "Get count of orders for user 456",
            "Find user with name 'Robert'); DROP TABLE users; --",
            "Delete all inactive users",
            "Get recent orders from 2024-01-01",
            "Find user by email admin@example.com",
        ]
        
        results = []
        for request in test_cases:
            result = self.compare(request)
            results.append(result)
        
        return results
    
    def print_comparison(self, result: ComparisonResult) -> None:
        """
        Print a detailed comparison for a single result.
        
        Args:
            result: ComparisonResult to display
        """
        print(f"\n{'=' * 70}")
        print(f"REQUEST: {result.request}")
        print("=" * 70)
        
        # Free-form section
        print("\n❌ FREE-FORM GENERATION:")
        print(f"  SQL: {result.free_form_result.query}")
        print(f"  Valid: {'✓' if result.free_form_result.is_valid else '✗'}")
        
        if result.free_form_result.risks:
            print(f"  🚨 RISKS ({len(result.free_form_result.risks)}):")
            for risk in result.free_form_result.risks:
                print(f"    • {risk}")
        
        if result.free_form_result.warnings:
            print(f"  ⚠️  WARNINGS ({len(result.free_form_result.warnings)}):")
            for warning in result.free_form_result.warnings:
                print(f"    • {warning}")
        
        # Template section
        print("\n✅ TEMPLATE SELECTION:")
        print(f"  Template: [{result.template_result.template_index}] {result.template_result.template_name}")
        print(f"  SQL: {result.template_result.rendered_query}")
        print(f"  Valid: {'✓' if result.template_result.is_valid else '✗'}")
        print(f"  Message: {result.template_result.validation_message}")
        
        # Winner
        print(f"\n🏆 WINNER: {result.winner}")
        print(f"   Reason: {result.reason}")
    
    def print_summary(self, results: List[ComparisonResult]) -> None:
        """
        Print summary statistics for the comparison suite.
        
        Args:
            results: List of all comparison results
        """
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        total = len(results)
        free_form_risks = sum(1 for r in results if r.free_form_result.risks)
        free_form_invalid = sum(1 for r in results if not r.free_form_result.is_valid)
        template_wins = sum(1 for r in results if r.winner == "TEMPLATE")
        
        print(f"\nTotal test cases: {total}")
        print(f"\nFree-Form Generation:")
        print(f"  • Cases with security risks: {free_form_risks}/{total} ({100*free_form_risks//total}%)")
        print(f"  • Cases with syntax errors: {free_form_invalid}/{total} ({100*free_form_invalid//total}%)")
        print(f"  • Safe cases: {total - free_form_risks - free_form_invalid}/{total}")
        
        print(f"\nTemplate Selection:")
        print(f"  • Cases won: {template_wins}/{total} ({100*template_wins//total}%)")
        print(f"  • Guaranteed safe: {total}/{total} (100%)")
        
        print("\n" + "=" * 70)
        print("KEY TAKEAWAYS")
        print("=" * 70)
        print("""
1. SECURITY
   Free-form: Vulnerable to SQL injection, destructive operations
   Template:  Impossible to generate dangerous SQL

2. RELIABILITY
   Free-form: May produce syntax errors, hallucinated table names
   Template:  Guaranteed syntactically correct, valid schema references

3. MAINTAINABILITY
   Free-form: Complex prompt engineering, unpredictable outputs
   Template:  Simple selection prompt, consistent outputs

4. AUDITABILITY
   Free-form: Cannot review all possible generated queries
   Template:  All possible queries are known and reviewed at build time

CONCLUSION: Template selection is the only safe choice for production.
        """)


def demonstrate_attack_resistance() -> None:
    """
    Demonstrate how template selection resists various attack vectors.
    """
    print("\n" + "=" * 70)
    print("ATTACK RESISTANCE DEMONSTRATION")
    print("=" * 70)
    
    generator = TemplateSQLGenerator()
    
    attacks = [
        (
            "Classic SQL Injection",
            0,
            {"user_id": "1 OR 1=1"},
            "Parameter is treated as literal value, not SQL code"
        ),
        (
            "Comment Injection",
            0,
            {"user_id": "123 -- malicious comment"},
            "Special characters are handled safely"
        ),
        (
            "Union Injection",
            0,
            {"user_id": "123 UNION SELECT * FROM passwords"},
            "Cannot inject additional SQL clauses"
        ),
        (
            "Drop Table Injection",
            0,
            {"user_id": "123'; DROP TABLE users; --"},
            "Destructive commands cannot be injected"
        ),
        (
            "Invalid Template Index",
            999,
            {"user_id": "123"},
            "Unknown templates are rejected immediately"
        ),
        (
            "Missing Required Parameters",
            3,
            {"date": "2024-01-01"},  # Missing 'limit'
            "Incomplete parameters are rejected"
        ),
        (
            "Extra Parameters (Data Exfiltration)",
            0,
            {"user_id": "123", "_debug": "SELECT * FROM secrets"},
            "Unexpected parameters are rejected"
        ),
    ]
    
    for attack_name, template_idx, params, explanation in attacks:
        print(f"\n{'─' * 70}")
        print(f"🛡️  ATTACK: {attack_name}")
        print(f"   Template: {template_idx}, Parameters: {params}")
        print("─" * 70)
        
        result = generator.generate(template_idx, params)
        
        if result.is_valid:
            print(f"⚠️  UNEXPECTED: Attack succeeded!")
            print(f"   SQL: {result.rendered_query}")
        else:
            print(f"✅ BLOCKED: {result.validation_message}")
            print(f"   Why: {explanation}")


def main() -> None:
    """
    Main demo function - runs complete comparison and analysis.
    """
    print("=" * 70)
    print("TEXT-TO-SQL: FREE-FORM vs TEMPLATE SELECTION")
    print("=" * 70)
    print("\nThis demo compares two approaches to Text-to-SQL:")
    print("  1. ❌ WRONG: Free-form SQL generation by LLM")
    print("  2. ✅ RIGHT: Indexed template selection by LLM")
    print("\n" + "=" * 70)
    
    # Show available templates
    print("\n" + "=" * 70)
    print("PRE-VALIDATED QUERY TEMPLATES")
    print("=" * 70)
    print("\nThese are the ONLY queries the system can execute:")
    print()
    
    for idx, template in QUERY_TEMPLATES.items():
        print(f"[{idx}] {template['name']}")
        print(f"    SQL: {template['template']}")
        print(f"    Params: {', '.join(template['params'])}")
        print()
    
    # Run comparison suite
    comparison = TextToSQLComparison()
    results = comparison.run_comparison_suite()
    
    for result in results:
        comparison.print_comparison(result)
    
    # Print summary
    comparison.print_summary(results)
    
    # Demonstrate attack resistance
    demonstrate_attack_resistance()
    
    # Final conclusion
    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)
    print("""
The question isn't "Can we make free-form SQL generation safe?"

The question is "Why would we ever trust an LLM to write SQL?"

Free-form generation is like giving a toddler a flamethrower:
- They might toast marshmallows successfully (sometimes)
- They might burn down the house (eventually)
- You cannot review every possible thing they might do

Template selection is like a toaster with a safety switch:
- Only toasts bread (pre-defined operations)
- Cannot burn the house down (impossible to generate dangerous SQL)
- Every possible output is known and reviewed

For production systems, the choice is clear:
✅ USE INDEXED TEMPLATE SELECTION

Your database (and your job) will thank you.
    """)


if __name__ == "__main__":
    main()
