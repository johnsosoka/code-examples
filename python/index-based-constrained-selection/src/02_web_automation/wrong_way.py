"""
WRONG WAY: Letting LLM generate XPath selectors.

This demonstrates the classic failure mode: LLMs hallucinate selectors
that look plausible but don't match the actual DOM structure.
"""

from typing import Optional
from .mock_page import WebElement, MOCK_PAGE


class XPathGenerator:
    """
    Simulates an LLM generating XPath selectors from HTML context.
    
    This is how NOT to do web automation - letting the LLM hallucinate
    selectors based on partial or outdated information.
    """
    
    def __init__(self):
        # Simulated "LLM responses" - showing common hallucination patterns
        self._llm_hallucinations = {
            "click login": [
                "//button[text()='Log In']",  # Wrong: text is 'Login', not 'Log In'
                "//div[@class='btn-primary']",  # Wrong: class doesn't exist
                "//button[@id='login']",  # Wrong: id is 'login-btn', not 'login'
            ],
            "enter username": [
                "//input[@placeholder='Enter username']",  # Wrong: no placeholder attr
                "//input[@id='user']",  # Wrong: id is different
                "//textbox[@name='username']",  # Wrong: textbox isn't a valid tag
            ],
            "click forgot password": [
                "//a[text()='Forgot password?']",  # Wrong: case and punctuation
                "//link[@href='/reset']",  # Wrong: link isn't a tag, href is '/reset-password'
            ],
        }
        self._attempt_count = 0
    
    def generate_xpath(self, instruction: str, html_context: str) -> str:
        """
        Simulate LLM generating an XPath.
        
        Returns a plausible-looking but potentially wrong selector.
        """
        # In reality, this would be an LLM call. We simulate common failures.
        instruction_lower = instruction.lower()
        
        # Get hallucinated options for this instruction type
        options = self._llm_hallucinations.get(instruction_lower, ["//unknown"])
        
        # Cycle through "creative" wrong answers
        xpath = options[self._attempt_count % len(options)]
        self._attempt_count += 1
        
        return xpath
    
    def find_element(self, xpath: str) -> Optional[WebElement]:
        """
        Try to find element by XPath.
        
        Returns None if XPath doesn't match (which happens often with LLM-generated selectors).
        """
        # Simulate XPath matching - in reality this would use browser automation
        for element in MOCK_PAGE:
            if element.xpath == xpath:
                return element
        return None
    
    def execute_action(self, instruction: str, html_context: str) -> dict:
        """
        Execute an action using LLM-generated XPath.
        
        Returns result showing success or failure.
        """
        # Step 1: LLM generates XPath
        xpath = self.generate_xpath(instruction, html_context)
        
        # Step 2: Try to find element
        element = self.find_element(xpath)
        
        if element is None:
            return {
                "success": False,
                "instruction": instruction,
                "generated_xpath": xpath,
                "error": "Element not found - XPath doesn't match any element",
                "element": None,
            }
        
        return {
            "success": True,
            "instruction": instruction,
            "generated_xpath": xpath,
            "element": element,
        }


def demonstrate_xpath_failures():
    """Run examples showing how XPath generation fails."""
    generator = XPathGenerator()
    
    # Simulate HTML context shown to LLM
    html_context = """
    <div class="login-form">
        <input name="username" />
        <input type="password" />
        <button id="login-btn">Login</button>
    </div>
    """
    
    instructions = [
        "click login",
        "enter username",
        "click forgot password",
    ]
    
    results = []
    for instruction in instructions:
        result = generator.execute_action(instruction, html_context)
        results.append(result)
    
    return results