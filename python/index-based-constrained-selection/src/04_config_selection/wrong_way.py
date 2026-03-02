"""
WRONG WAY: Free-form LLM configuration generation.

This approach asks the LLM to generate configuration YAML directly.
Problems:
- LLM may hallucinate invalid model names (gpt-5, gpt-4-turbo-ultra)
- Values may be out of valid ranges (temperature: 5.0)
- Syntax errors in YAML
- Missing required fields
- Inconsistent formatting

This is like asking someone to write a prescription without a formulary.
"""

import sys
from pathlib import Path
from typing import Any

# Handle imports for both module and script usage
try:
    from .config_models import LLMConfig
except ImportError:
    # Running as script
    sys.path.insert(0, str(Path(__file__).parent))
    from config_models import LLMConfig

import yaml
from pydantic import ValidationError


class WrongWayConfigSelector:
    """
    Demonstrates the problematic approach of letting LLMs generate configs freely.
    
    This simulates what happens when you ask an LLM to "generate a config"
    without constraints. The results are... unpredictable.
    """
    
    def __init__(self):
        """Initialize the wrong-way selector."""
        self.validation_errors: list[dict[str, Any]] = []
    
    def simulate_llm_response(self, prompt: str) -> str:
        """
        Simulate various problematic LLM responses to config generation prompts.
        
        In reality, this would be an actual LLM call. Here we simulate
        realistic failure modes that occur with free-form generation.
        
        Args:
            prompt: The configuration request prompt
            
        Returns:
            Simulated YAML response from LLM
        """
        # Simulate different problematic responses based on prompt content
        if "production" in prompt.lower():
            # LLM tries to be helpful but hallucinates model name
            return """
model: gpt-5-pro
max_tokens: 1000
temperature: 0.0
timeout: 30
retries: 3
"""
        elif "creative" in prompt.lower():
            # LLM uses invalid temperature value
            return """
model: gpt-4
temperature: 5.0
max_tokens: 2000
timeout: 60
retries: 2
"""
        elif "fast" in prompt.lower():
            # LLM omits required fields
            return """
model: gpt-3.5-turbo
temperature: 0.0
max_tokens: 500
"""
        elif "reliable" in prompt.lower():
            # LLM uses out-of-range values
            return """
model: gpt-4
temperature: 0.0
max_tokens: 10000
timeout: 600
retries: 10
"""
        else:
            # Default: syntax error in YAML
            return """
model: gpt-4
temperature: 0.7
max_tokens: 1500
timeout: 45
retries: 3
extra_field: this_should_not_be_here
"""
    
    def parse_config(self, yaml_content: str) -> tuple[LLMConfig | None, list[str]]:
        """
        Attempt to parse YAML and validate against LLMConfig.
        
        Args:
            yaml_content: Raw YAML string from LLM
            
        Returns:
            Tuple of (parsed config or None, list of error messages)
        """
        errors: list[str] = []
        
        # Step 1: Parse YAML
        try:
            raw_config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {e}")
            return None, errors
        
        if not isinstance(raw_config, dict):
            errors.append(f"Expected YAML mapping, got {type(raw_config).__name__}")
            return None, errors
        
        # Step 2: Validate with Pydantic
        try:
            config = LLMConfig(**raw_config)
            return config, errors
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                errors.append(f"Validation error on '{field}': {msg}")
            return None, errors
    
    def generate_config(self, prompt: str) -> dict[str, Any]:
        """
        Generate configuration using free-form LLM approach.
        
        Args:
            prompt: Description of desired configuration
            
        Returns:
            Dictionary with results and any errors
        """
        print(f"\n{'='*60}")
        print(f"WRONG WAY: Free-form generation")
        print(f"{'='*60}")
        print(f"\nPrompt: '{prompt}'")
        
        # Simulate LLM generating config
        raw_yaml = self.simulate_llm_response(prompt)
        print(f"\n🤖 LLM generated YAML:")
        print("-" * 40)
        print(raw_yaml.strip())
        print("-" * 40)
        
        # Attempt to parse and validate
        config, errors = self.parse_config(raw_yaml)
        
        result = {
            "prompt": prompt,
            "raw_yaml": raw_yaml,
            "success": config is not None,
            "config": config,
            "errors": errors
        }
        
        if config:
            print(f"\n✅ Config parsed successfully!")
            print(f"   Model: {config.model}")
            print(f"   Temperature: {config.temperature}")
            print(f"   Max tokens: {config.max_tokens}")
        else:
            print(f"\n❌ Validation failed with {len(errors)} error(s):")
            for i, error in enumerate(errors, 1):
                print(f"   {i}. {error}")
                self.validation_errors.append({
                    "prompt": prompt,
                    "error": error
                })
        
        return result
    
    def demonstrate_failure_modes(self) -> list[dict[str, Any]]:
        """
        Run through various failure modes to show why this approach is problematic.
        
        Returns:
            List of results from each test case
        """
        print("\n" + "="*60)
        print("DEMONSTRATING FREE-FORM CONFIG GENERATION FAILURES")
        print("="*60)
        
        test_cases = [
            "Generate LLM config for a production system",
            "Generate LLM config for creative writing",
            "Generate LLM config for fast responses",
            "Generate LLM config for maximum reliability",
            "Generate LLM config with default settings",
        ]
        
        results = []
        for prompt in test_cases:
            result = self.generate_config(prompt)
            results.append(result)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        print(f"Total attempts: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        
        if failed > 0:
            print(f"\n⚠️  {failed} configuration(s) failed validation!")
            print("   In production, these would cause runtime errors.")
        
        return results