"""
RIGHT WAY: Index-based element selection.

Instead of letting LLM hallucinate XPaths, we:
1. Discover actual elements from the DOM
2. Present them as an indexed list
3. LLM selects by index (deterministic)
4. System retrieves the exact, validated XPath
"""

from ..common.registry import IndexedRegistry
from ..common.models import ElementSelection
from .mock_page import WebElement, MOCK_PAGE


class IndexedPageController:
    """
    Controller that uses index-based selection for web automation.
    
    The key insight: LLM never sees or generates XPaths.
    It only selects from a numbered list of discovered elements.
    """
    
    def __init__(self, elements: list[WebElement]):
        # Create registry with discovered elements
        # The describer controls what the LLM sees (no XPaths!)
        self.registry = IndexedRegistry(
            items={i: elem for i, elem in enumerate(elements)},
            describer=lambda elem: f"{elem.role}: \"{elem.name}\""
        )
    
    def get_accessibility_tree(self) -> str:
        """
        Generate accessibility tree format for LLM context.
        
        This is what the LLM sees - indices and descriptions only.
        No XPaths, no HTML structure, no room for hallucination.
        """
        lines = []
        for idx, desc in self.registry.get_context().items():
            lines.append(f"[{idx}] {desc}")
        return "\n".join(lines)
    
    def select_element(self, instruction: str) -> ElementSelection:
        """
        Simulate LLM selecting an element by index.
        
        In reality, this would be an LLM call with the accessibility tree
        in the context and ElementSelection as the structured output.
        """
        # Simulate LLM reasoning based on instruction
        instruction_lower = instruction.lower()
        
        # Simple keyword matching to simulate LLM selection logic
        # Check more specific patterns first!
        if "forgot" in instruction_lower:
            selected = 3
            reasoning = "Forgot Password link is element [3]"
        elif "login" in instruction_lower:
            selected = 0
            reasoning = "Login button is element [0]"
        elif "username" in instruction_lower:
            selected = 1
            reasoning = "Username field is element [1]"
        elif "password" in instruction_lower:
            selected = 2
            reasoning = "Password field is element [2]"
        elif "sign up" in instruction_lower or "signup" in instruction_lower:
            selected = 4
            reasoning = "Sign Up button is element [4]"
        else:
            selected = 0
            reasoning = "Defaulting to first element"
        
        # Determine action from instruction
        if "click" in instruction_lower:
            action = "click"
        elif "type" in instruction_lower or "enter" in instruction_lower:
            action = "type"
        else:
            action = "click"
        
        return ElementSelection(
            selected_index=selected,
            action=action,
            reasoning=reasoning
        )
    
    def execute_action(self, instruction: str) -> dict:
        """
        Execute an action using index-based selection.
        
        This is deterministic - the index maps to a pre-validated element.
        """
        # Step 1: Get LLM selection
        selection = self.select_element(instruction)
        
        # Step 2: Retrieve the actual element (guaranteed to exist)
        try:
            element = self.registry.get(selection.selected_index)
        except IndexError as e:
            return {
                "success": False,
                "instruction": instruction,
                "error": str(e),
            }
        
        return {
            "success": True,
            "instruction": instruction,
            "selected_index": selection.selected_index,
            "action": selection.action,
            "reasoning": selection.reasoning,
            "element": element,
            "xpath_used": element.xpath,  # System retrieves this, not LLM
        }


def demonstrate_index_selection():
    """Run examples showing index-based selection works."""
    controller = IndexedPageController(MOCK_PAGE)
    
    # Show what the LLM sees
    print("Accessibility Tree (what LLM sees):")
    print(controller.get_accessibility_tree())
    print()
    
    instructions = [
        "click login",
        "enter username",
        "click forgot password",
    ]
    
    results = []
    for instruction in instructions:
        result = controller.execute_action(instruction)
        results.append(result)
    
    return results