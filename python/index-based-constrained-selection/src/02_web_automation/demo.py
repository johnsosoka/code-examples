"""
Demo: Compare XPath generation (WRONG) vs Index selection (RIGHT).

Run this to see the dramatic difference in reliability.
"""

from .wrong_way import demonstrate_xpath_failures
from .right_way import demonstrate_index_selection, IndexedPageController
from .mock_page import MOCK_PAGE


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_comparison():
    """Run full comparison of both approaches."""
    
    print_section("WEB AUTOMATION: WRONG vs RIGHT WAY")
    
    # Show the mock page
    print("\n📄 MOCK PAGE ELEMENTS:")
    print("-" * 40)
    for i, elem in enumerate(MOCK_PAGE):
        print(f"  [{i}] {elem.role}: '{elem.name}'")
        print(f"      XPath: {elem.xpath}")
    
    # WRONG WAY
    print_section("❌ WRONG WAY: LLM Generates XPath")
    print("\nApproach: Show HTML to LLM, ask it to generate XPath")
    print("Problem: LLM hallucinates selectors that don't exist\n")
    
    xpath_results = demonstrate_xpath_failures()
    
    for result in xpath_results:
        print(f"Instruction: '{result['instruction']}'")
        print(f"  Generated XPath: {result['generated_xpath']}")
        if result['success']:
            print(f"  ✅ Found: {result['element'].name}")
        else:
            print(f"  ❌ FAILED: {result['error']}")
        print()
    
    # RIGHT WAY
    print_section("✅ RIGHT WAY: Index-Based Selection")
    print("\nApproach: Present indexed list, LLM selects by number")
    print("Benefit: Deterministic mapping to pre-validated elements\n")
    
    # Show accessibility tree
    controller = IndexedPageController(MOCK_PAGE)
    print("Accessibility Tree (sent to LLM):")
    print("-" * 40)
    print(controller.get_accessibility_tree())
    print()
    
    index_results = demonstrate_index_selection()
    
    for result in index_results:
        print(f"Instruction: '{result['instruction']}'")
        print(f"  Selected Index: {result['selected_index']}")
        print(f"  Action: {result['action']}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Resolved XPath: {result['xpath_used']}")
        print(f"  ✅ SUCCESS: {result['element'].name}")
        print()
    
    # Summary
    print_section("COMPARISON SUMMARY")
    
    xpath_success = sum(1 for r in xpath_results if r['success'])
    index_success = sum(1 for r in index_results if r['success'])
    
    print(f"\n{'Approach':<25} {'Success Rate':>15} {'Failure Mode':<30}")
    print("-" * 70)
    print(f"{'XPath Generation':<25} {f'{xpath_success}/{len(xpath_results)}':>15} {'Hallucinated selectors':<30}")
    print(f"{'Index Selection':<25} {f'{index_success}/{len(index_results)}':>15} {'None (deterministic)':<30}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("  XPath generation fails because LLMs don't actually see the DOM.")
    print("  They guess based on patterns, and guesses are often wrong.")
    print()
    print("  Index selection works because the system controls the mapping.")
    print("  LLM only picks a number; system retrieves the validated XPath.")
    print("=" * 60)


if __name__ == "__main__":
    run_comparison()