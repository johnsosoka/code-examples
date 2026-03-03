"""
Demo: Configuration Selection - Free-Form vs Validated Templates

This demo compares two approaches to LLM configuration:

WRONG WAY (wrong_way.py):
  - LLM generates YAML configuration free-form
  - Prone to hallucinations, invalid values, syntax errors
  - Requires runtime validation that often fails
  - Unpredictable behavior in production

RIGHT WAY (right_way.py):
  - Pre-validated configuration templates
  - LLM selects by index based on use case
  - All configs guaranteed valid at system startup
  - Deterministic, reliable behavior

Key Insight:
  LLMs excel at REASONING ("which config fits this use case?")
  but are unreliable at GENERATION ("write me a valid config").
  
  By separating concerns:
  1. Humans define validated templates (with constraints)
  2. LLM selects the appropriate template (using reasoning)
  3. System guarantees validity (no runtime surprises)
"""

from typing import Any

from .wrong_way import WrongWayConfigSelector
from .right_way import RightWayConfigSelector


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_comparison_table() -> None:
    """Print a comparison table of the two approaches."""
    print_section_header("COMPARISON: FREE-FORM VS INDEX-BASED")
    
    print("""
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Aspect              │ FREE-FORM (Wrong Way)    │ INDEX-BASED (Right Way)  │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ LLM Task            │ Generate YAML            │ Select index             │
│ Validation          │ Runtime (often fails)    │ At system startup        │
│ Error Rate          │ High (20-40%)            │ Zero (if index valid)    │
│ Model Hallucination │ Common (gpt-5, etc.)     │ Impossible               │
│ Out-of-Range Values │ Common (temp: 5.0)       │ Impossible               │
│ Missing Fields      │ Common                   │ Impossible               │
│ YAML Syntax Errors  │ Occasional               │ N/A (no YAML parsing)    │
│ Performance         │ Slow (parse + validate)  │ Fast (dict lookup)       │
│ Maintainability     │ Hard (handle errors)     │ Easy (pre-validated)     │
│ Production Safety   │ Risky                    │ Guaranteed               │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
""")


def run_wrong_way_demo() -> list[dict[str, Any]]:
    """Run the free-form configuration generation demo."""
    print_section_header("PART 1: WRONG WAY - FREE-FORM GENERATION")

    selector = WrongWayConfigSelector()
    results = selector.demonstrate_failure_modes()

    return results


def run_right_way_demo() -> list[dict[str, Any]]:
    """Run the index-based configuration selection demo."""
    print_section_header("PART 2: RIGHT WAY - INDEX-BASED SELECTION")

    selector = RightWayConfigSelector()
    results = selector.demonstrate_selection()

    return results


def print_key_takeaways() -> None:
    """Print key takeaways from the demo."""
    print_section_header("KEY TAKEAWAYS")
    
    print("""
1. SEPARATE CONCERNS
   • Let humans define WHAT configurations are valid (with constraints)
   • Let LLMs reason about WHICH configuration fits the use case
   • Let the system guarantee validity through structure

2. LLM STRENGTHS VS WEAKNESSES
   ✅ LLMs excel at: Reasoning, comparison, selection, explanation
   ❌ LLMs struggle with: Precise syntax, valid ranges, constraint satisfaction

3. PRE-VALIDATION PATTERN
   • Define valid options upfront with Pydantic models
   • Register them in an IndexedRegistry
   • Present options to LLM with descriptions
   • LLM returns only an index
   • System retrieves pre-validated config

4. PRODUCTION BENEFITS
   • Zero runtime validation surprises
   • Deterministic behavior
   • Faster (no parsing/validation on hot path)
   • Easier to debug (clear selection trail)
   • Safer deployments

5. REAL-WORLD ANALOGY
   Free-form: "Describe how to get to the airport"
   Index-based: "Select your route: [0] Highway [1] Back roads [2] Transit"
   
   The second approach eliminates "turn left at the big tree" ambiguity.
""")


def main() -> None:
    """Run the complete configuration selection demo."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     CONFIGURATION SELECTION: FREE-FORM VS VALIDATED TEMPLATES        ║
║                                                                      ║
║     A demonstration of why index-based selection beats               ║
║     free-form generation for LLM configurations                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Part 1: Wrong Way
    wrong_results = run_wrong_way_demo()
    
    # Part 2: Right Way
    right_results = run_right_way_demo()
    
    # Comparison
    print_comparison_table()
    
    # Takeaways
    print_key_takeaways()
    
    # Final summary
    print_section_header("FINAL SUMMARY")
    
    wrong_success = sum(1 for r in wrong_results if r["success"])
    right_success = sum(1 for r in right_results if r["success"])
    
    print(f"""
Results from this demo:

  Free-form generation:  {wrong_success}/{len(wrong_results)} successful ({wrong_success/len(wrong_results)*100:.0f}%)
  Index-based selection: {right_success}/{len(right_results)} successful ({right_success/len(right_results)*100:.0f}%)

The index-based approach:
  ✓ Eliminates validation errors
  ✓ Prevents hallucinated values
  ✓ Guarantees constraints are satisfied
  ✓ Faster and more reliable

Use this pattern whenever you need LLMs to work with structured data
that has strict validity constraints.
""")


if __name__ == "__main__":
    main()