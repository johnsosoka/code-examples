"""
RIGHT WAY: Index-based configuration template selection.

This approach presents pre-validated configuration templates to the LLM
and asks it to SELECT which one to use by index.

Benefits:
- ALL templates are validated at registration time
- LLM only needs to reason about which config fits the use case
- Impossible to generate invalid configurations
- No YAML parsing needed
- Fast, deterministic selection

This is like giving someone a menu instead of asking them to cook.
"""

import sys
from typing import Any
from pathlib import Path

# Handle imports for both module and script usage
try:
    from common.registry import IndexedRegistry
    from common.models import ConfigSelection
    from .config_models import LLMConfig, CONFIG_TEMPLATES, get_template_descriptions
except ImportError:
    # Running as script
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from common.registry import IndexedRegistry
    from common.models import ConfigSelection
    from config_models import LLMConfig, CONFIG_TEMPLATES, get_template_descriptions


class RightWayConfigSelector:
    """
    Demonstrates the correct approach: pre-validated template selection.
    
    Configuration templates are validated ONCE at system initialization.
    The LLM only selects which pre-validated template to use.
    """
    
    def __init__(self):
        """Initialize the registry with pre-validated config templates."""
        # Create indexed registry from templates
        # These are GUARANTEED valid - they were validated at import time
        self.registry = IndexedRegistry(CONFIG_TEMPLATES)
        self.selection_history: list[dict[str, Any]] = []
        
        print("📝 Pre-validated configuration templates loaded:")
        for idx, template in CONFIG_TEMPLATES.items():
            config = template["config"]
            print(f"   [{idx}] {template['name']}: {template['description']}")
            print(f"       → {config.model}, temp={config.temperature}, "
                  f"tokens={config.max_tokens}")
    
    def build_selection_prompt(self, use_case: str) -> str:
        """
        Build a prompt that presents templates and asks for selection.
        
        Args:
            use_case: Description of what the config will be used for
            
        Returns:
            Formatted prompt string
        """
        descriptions = get_template_descriptions()
        
        prompt_lines = [
            "Select the most appropriate LLM configuration for this use case:",
            f"\nUse case: {use_case}\n",
            "Available configurations:",
        ]
        
        for idx in sorted(descriptions.keys()):
            prompt_lines.append(f"  {descriptions[idx]}")
        
        prompt_lines.extend([
            "\nRespond with ONLY a JSON object in this exact format:",
            '{"config_index": <number>, "reasoning": "<brief explanation>"}'
        ])
        
        return "\n".join(prompt_lines)
    
    def simulate_llm_selection(self, use_case: str) -> ConfigSelection:
        """
        Simulate LLM selecting the appropriate config template.
        
        In reality, this would be an actual LLM call with the prompt.
        The LLM reasons about which config fits best and returns an index.
        
        Args:
            use_case: Description of the use case
            
        Returns:
            ConfigSelection with chosen index and reasoning
        """
        # Simulate intelligent selection based on use case
        use_case_lower = use_case.lower()
        
        if any(word in use_case_lower for word in ["production", "conservative", "safe"]):
            return ConfigSelection(
                config_index=0,
                reasoning="Production systems require conservative, deterministic settings with gpt-4"
            )
        elif any(word in use_case_lower for word in ["creative", "writing", "story", "brainstorm"]):
            return ConfigSelection(
                config_index=1,
                reasoning="Creative tasks benefit from higher temperature (0.8) for varied outputs"
            )
        elif any(word in use_case_lower for word in ["fast", "quick", "cheap", "latency"]):
            return ConfigSelection(
                config_index=2,
                reasoning="Fast responses prioritized with cheaper gpt-3.5-turbo model"
            )
        elif any(word in use_case_lower for word in ["reliable", "critical", "important", "must not fail"]):
            return ConfigSelection(
                config_index=3,
                reasoning="Maximum reliability requires gpt-4 with maximum retries (5)"
            )
        else:
            # Default to production settings for unknown cases
            return ConfigSelection(
                config_index=0,
                reasoning="Defaulting to conservative production settings for safety"
            )
    
    def select_config(self, use_case: str) -> dict[str, Any]:
        """
        Select configuration using index-based approach.
        
        Args:
            use_case: Description of desired configuration use case
            
        Returns:
            Dictionary with selected config and metadata
        """
        print(f"\n{'='*60}")
        print(f"RIGHT WAY: Index-based selection")
        print(f"{'='*60}")
        print(f"\nUse case: '{use_case}'")
        
        # Build and show the prompt
        prompt = self.build_selection_prompt(use_case)
        print(f"\n📋 Prompt sent to LLM:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
        
        # Get LLM selection
        selection = self.simulate_llm_selection(use_case)
        print(f"\n🤖 LLM response:")
        print(f"   Selected index: {selection.config_index}")
        print(f"   Reasoning: {selection.reasoning}")
        
        # Validate index is in range
        if selection.config_index not in self.registry:
            error_msg = f"Invalid index {selection.config_index}. Valid: {list(self.registry._items.keys())}"
            print(f"\n❌ Error: {error_msg}")
            return {
                "use_case": use_case,
                "success": False,
                "error": error_msg,
                "selection": selection
            }
        
        # Retrieve pre-validated config
        template = self.registry.get(selection.config_index)
        config = template["config"]
        
        print(f"\n✅ Configuration retrieved:")
        print(f"   Name: {template['name']}")
        print(f"   Model: {config.model}")
        print(f"   Temperature: {config.temperature}")
        print(f"   Max tokens: {config.max_tokens}")
        print(f"   Timeout: {config.timeout}s")
        print(f"   Retries: {config.retries}")
        print(f"\n✨ GUARANTEED VALID - validated at system startup")
        
        result = {
            "use_case": use_case,
            "success": True,
            "selection": selection,
            "template_name": template["name"],
            "config": config
        }
        
        self.selection_history.append(result)
        return result
    
    def demonstrate_selection(self) -> list[dict[str, Any]]:
        """
        Run through various use cases to show reliable selection.
        
        Returns:
            List of results from each test case
        """
        print("\n" + "="*60)
        print("DEMONSTRATING INDEX-BASED CONFIG SELECTION")
        print("="*60)
        
        test_cases = [
            "Production API that needs conservative, safe responses",
            "Creative writing assistant for story generation",
            "Fast chatbot with minimal latency requirements",
            "Critical medical diagnosis support system",
            "General purpose assistant with default settings",
        ]
        
        results = []
        for use_case in test_cases:
            result = self.select_config(use_case)
            results.append(result)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        successful = sum(1 for r in results if r["success"])
        print(f"Total selections: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        print(f"\n✨ All {len(results)} configurations are pre-validated!")
        print("   Zero runtime validation errors possible.")
        
        return results
    
    def get_config_by_index(self, index: int) -> LLMConfig | None:
        """
        Directly retrieve a config by index (for programmatic use).
        
        Args:
            index: Template index
            
        Returns:
            LLMConfig or None if index invalid
        """
        template = self.registry.get(index)
        if template:
            return template["config"]
        return None