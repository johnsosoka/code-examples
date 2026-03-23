"""Mock web page elements for demonstration."""

from dataclasses import dataclass


@dataclass
class WebElement:
    """Simplified web element representation."""
    role: str
    name: str
    xpath: str  # The actual selector - hidden from LLM in right_way


# A simple mock page with 5 elements
# In real automation, these would be discovered from the actual DOM
MOCK_PAGE = [
    WebElement(
        role="button",
        name="Login",
        xpath="//button[@id='login-btn']"
    ),
    WebElement(
        role="textbox",
        name="Username",
        xpath="//input[@name='username']"
    ),
    WebElement(
        role="textbox",
        name="Password",
        xpath="//input[@type='password']"
    ),
    WebElement(
        role="link",
        name="Forgot Password",
        xpath="//a[@href='/reset-password']"
    ),
    WebElement(
        role="button",
        name="Sign Up",
        xpath="//button[contains(@class, 'signup')]"
    ),
]