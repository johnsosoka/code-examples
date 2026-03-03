"""Web automation example: XPath generation vs index-based selection."""

from .mock_page import WebElement, MOCK_PAGE
from .wrong_way import XPathGenerator, demonstrate_xpath_failures
from .right_way import IndexedPageController, demonstrate_index_selection
from .demo import run_comparison

__all__ = [
    "WebElement",
    "MOCK_PAGE",
    "XPathGenerator",
    "demonstrate_xpath_failures",
    "IndexedPageController",
    "demonstrate_index_selection",
    "run_comparison",
]