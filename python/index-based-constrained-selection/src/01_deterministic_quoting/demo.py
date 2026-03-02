"""
Demo: Deterministic Quoting with Sentence-Level Indexing

This script demonstrates the difference between:
- WRONG way: Free-form quote generation (LLM writes quotes directly)
- RIGHT way: Index-based selection (LLM selects sentence indices)

Run this script to see a side-by-side comparison showing why
the index-based approach guarantees perfect accuracy.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrong_way import extract_quotes_wrong_way, verify_quotes_accuracy
from right_way import extract_quotes_right_way, verify_quotes_deterministic


def load_sample_essay() -> str:
    """Load the sample essay from the file."""
    essay_path = Path(__file__).parent / "sample_essay.txt"
    with open(essay_path, "r") as f:
        return f.read()


def print_section_header(title: str, width: int = 80) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")


def print_separator(char: str = "-", width: int = 80) -> None:
    """Print a separator line."""
    print(char * width)


def run_wrong_way_demo(essay_text: str, topic: str) -> None:
    """Run the WRONG way demo: free-form quote generation.
    
    This demonstrates the problematic approach where we ask the LLM
to generate quotes directly, which can lead to hallucinations
    and inaccuracies.
    
    Args:
        essay_text: The full essay text.
        topic: The topic to extract quotes about.
    """
    print_section_header("WRONG WAY: Free-Form Quote Generation")
    
    print("Approach: Ask LLM to generate quotes directly")
    print("Prompt: 'Extract relevant quotes about [topic] from this essay'")
    print_separator()
    
    try:
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  OPENAI_API_KEY not set. Using mock response for demonstration.\n")
            # Mock response for demo purposes
            from wrong_way import FreeFormQuotes
            result = FreeFormQuotes(
                quotes=[
                    "Artificial intelligence has emerged as one of the most transformative technologies in modern healthcare.",
                    "Machine learning algorithms can now analyze medical images with superhuman precision.",
                    "The question of accountability stands at the forefront of these ethical concerns."
                ],
                reasoning="These quotes capture the main themes of AI in healthcare and the ethical challenges discussed."
            )
        else:
            result = extract_quotes_wrong_way(essay_text, topic)
        
        print(f"\n📊 LLM Reasoning:")
        print(f"   {result.reasoning}\n")
        
        print(f"📝 Generated Quotes ({len(result.quotes)} found):")
        for i, quote in enumerate(result.quotes, 1):
            print(f"\n   [{i}] {quote}")
        
        # Verify accuracy
        print("\n🔍 Accuracy Verification:")
        verification = verify_quotes_accuracy(essay_text, result.quotes)
        
        print(f"   Total quotes: {verification['total_quotes']}")
        print(f"   Exact matches: {verification['exact_matches']}")
        print(f"   Partial matches: {verification['partial_matches']}")
        print(f"   Not found: {verification['not_found']}")
        print(f"   Accuracy score: {verification['accuracy_score']:.1f}%")
        
        # Show details
        print("\n   Detailed Results:")
        for detail in verification['details']:
            status_icon = "✅" if detail['status'] == 'exact_match' else "⚠️" if detail['status'] == 'partial_match' else "❌"
            print(f"   {status_icon} {detail['status'].upper()}: {detail['quote']}")
        
        print_separator()
        print("\n⚠️  PROBLEMS WITH THIS APPROACH:")
        print("   1. LLM may paraphrase instead of quoting exactly")
        print("   2. Words may be added, removed, or changed")
        print("   3. Multiple sentences may be merged incorrectly")
        print("   4. Cannot guarantee the quote exists in the original")
        print("   5. No way to verify accuracy programmatically")
        
    except Exception as e:
        print(f"❌ Error in wrong_way demo: {e}")
        import traceback
        traceback.print_exc()


def run_right_way_demo(essay_text: str, topic: str) -> None:
    """Run the RIGHT way demo: index-based selection.
    
    This demonstrates the correct approach where we:
    1. Split text into indexed sentences
    2. Ask LLM to select indices
    3. Retrieve exact sentences deterministically
    
    Args:
        essay_text: The full essay text.
        topic: The topic to extract quotes about.
    """
    print_section_header("RIGHT WAY: Index-Based Sentence Selection")
    
    print("Approach: Split into sentences, ask LLM to select indices, retrieve exactly")
    print("Prompt: 'Select the indices of sentences relevant to [topic]'")
    print_separator()
    
    try:
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  OPENAI_API_KEY not set. Using mock response for demonstration.\n")
            # Mock response for demo purposes
            from right_way import IndexedSentenceRegistry, QuoteSelection
            registry = IndexedSentenceRegistry(essay_text)
            
            # Simulate LLM selecting some indices
            mock_selection = QuoteSelection(
                selected_indices=[0, 2, 7],
                reasoning="Sentences 0 and 2 discuss AI's transformative impact on healthcare. Sentence 7 addresses accountability concerns."
            )
            
            result = {
                "quotes": registry.get_sentences_by_indices(mock_selection.selected_indices),
                "selected_indices": mock_selection.selected_indices,
                "reasoning": mock_selection.reasoning,
                "registry": registry
            }
        else:
            result = extract_quotes_right_way(essay_text, topic)
            registry = result["registry"]
        
        print(f"\n📊 LLM Reasoning:")
        print(f"   {result['reasoning']}\n")
        
        print(f"🔢 Selected Indices: {result['selected_indices']}\n")
        
        print(f"📝 Retrieved Quotes ({len(result['quotes'])} found):")
        for i, (idx, quote) in enumerate(zip(result['selected_indices'], result['quotes']), 1):
            print(f"\n   [{i}] Index [{idx}]: {quote}")
        
        # Verify accuracy
        print("\n🔍 Accuracy Verification:")
        verification = verify_quotes_deterministic(
            essay_text, 
            result['quotes'], 
            result['registry']
        )
        
        print(f"   Total quotes: {verification['total_quotes']}")
        print(f"   Exact matches: {verification['exact_matches']} ✅")
        print(f"   All in registry: {verification['all_in_registry']} ✅")
        print(f"   Accuracy score: {verification['accuracy_score']:.1f}% ✅")
        
        # Show details
        print("\n   Detailed Results:")
        for detail in verification['details']:
            status_icon = "✅" if detail['status'] == 'exact_match' else "❌"
            print(f"   {status_icon} {detail['status'].upper()}")
            print(f"      In registry: {detail['in_registry']}")
            print(f"      In original: {detail['in_original']}")
            print(f"      Quote: {detail['quote']}")
        
        print_separator()
        print("\n✅ BENEFITS OF THIS APPROACH:")
        print("   1. Guaranteed exact quotes from original text")
        print("   2. No possibility of hallucination or modification")
        print("   3. Verifiable programmatically")
        print("   4. Deterministic - same indices always return same text")
        print("   5. Audit trail: you know exactly which sentences were selected")
        
    except Exception as e:
        print(f"❌ Error in right_way demo: {e}")
        import traceback
        traceback.print_exc()


def print_comparison_summary() -> None:
    """Print a summary comparing the two approaches."""
    print_section_header("COMPARISON SUMMARY")
    
    print("┌─────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Aspect              │ WRONG WAY            │ RIGHT WAY            │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┤")
    print("│ LLM Task            │ Generate quotes      │ Select indices       │")
    print("│ Accuracy            │ Variable (~80-95%)   │ Guaranteed (100%)    │")
    print("│ Verifiable          │ ❌ Difficult         │ ✅ Trivial           │")
    print("│ Hallucination Risk  │ ⚠️ High              │ ✅ None              │")
    print("│ Audit Trail         │ ❌ None              │ ✅ Full (indices)    │")
    print("│ Deterministic       │ ❌ No                │ ✅ Yes               │")
    print("│ Reproducible        │ ❌ Unreliable        │ ✅ Always            │")
    print("└─────────────────────┴──────────────────────┴──────────────────────┘")
    
    print("\n🎯 KEY INSIGHT:")
    print("   When you need exact, verifiable quotes from source text:")
    print("   → NEVER ask the LLM to generate the text")
    print("   → ALWAYS ask the LLM to select indices/identifiers")
    print("   → Then retrieve the text deterministically")
    
    print("\n💡 This pattern applies beyond quoting:")
    print("   - Citation extraction (select reference IDs)")
    print("   - Document classification (select category IDs)")
    print("   - Entity extraction (select from predefined list)")
    print("   - Any task requiring exact, verifiable output from a set")


def main() -> None:
    """Main entry point for the demo."""
    print_section_header("DETERMINISTIC QUOTING DEMO")
    print("Comparing: Free-Form Generation vs Index-Based Selection")
    print(f"Topic: Artificial intelligence ethics in healthcare")
    
    # Load the essay
    essay_text = load_sample_essay()
    topic = "artificial intelligence ethics in healthcare"
    
    print(f"\n📄 Loaded essay: {len(essay_text)} characters")
    print(f"🎯 Topic: {topic}")
    
    # Run both demos
    run_wrong_way_demo(essay_text, topic)
    run_right_way_demo(essay_text, topic)
    
    # Print comparison
    print_comparison_summary()
    
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()