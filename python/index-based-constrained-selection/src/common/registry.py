"""Generic indexed registry pattern for constrained selection."""

from collections.abc import Callable, ItemsView
from typing import TypeVar, Generic

T = TypeVar("T")


class IndexedRegistry(Generic[T]):
    """
    A registry of pre-validated items with index-based selection.
    
    This implements the core pattern: instead of generating solutions,
    the LLM selects from pre-validated options by index.
    
    Example:
        >>> registry = IndexedRegistry({
        ...     0: "Option A",
        ...     1: "Option B",
        ... })
        >>> registry.get(0)
        'Option A'
    """
    
    def __init__(
        self,
        items: dict[int, T],
        validator: Callable[[T], bool] | None = None,
        describer: Callable[[T], str] | None = None,
    ):
        """
        Initialize registry with pre-validated items.
        
        Args:
            items: Dictionary mapping indices to items
            validator: Optional function to validate items
            describer: Optional function to generate descriptions for LLM context
        """
        self._items = items
        self._validator = validator or (lambda x: True)
        self._describer = describer or str
        self._validate_all()
    
    def _validate_all(self) -> None:
        """Validate all items at registration time."""
        for idx, item in self._items.items():
            if not self._validator(item):
                raise ValueError(f"Item at index {idx} failed validation: {item}")
    
    def get(self, index: int) -> T:
        """
        Deterministically retrieve item by index.
        
        Args:
            index: The index to retrieve
            
        Returns:
            The item at that index
            
        Raises:
            IndexError: If index not in registry
        """
        if index not in self._items:
            valid = list(self._items.keys())
            raise IndexError(f"Invalid index: {index}. Valid indices: {valid}")
        return self._items[index]
    
    def get_context(self) -> dict[int, str]:
        """
        Get LLM-friendly context mapping indices to descriptions.
        
        Returns:
            Dictionary of index -> description
        """
        return {idx: self._describer(item) for idx, item in self._items.items()}
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __contains__(self, index: int) -> bool:
        return index in self._items
    
    def items(self) -> ItemsView[int, T]:
        return self._items.items()
