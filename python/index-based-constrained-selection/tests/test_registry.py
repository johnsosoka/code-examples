"""Tests for the IndexedRegistry pattern."""

import pytest
from src.common.registry import IndexedRegistry


class TestIndexedRegistry:
    """Test suite for IndexedRegistry."""
    
    def test_basic_retrieval(self):
        """Test basic index-based retrieval."""
        registry = IndexedRegistry({
            0: "item_a",
            1: "item_b",
            2: "item_c",
        })
        
        assert registry.get(0) == "item_a"
        assert registry.get(1) == "item_b"
        assert registry.get(2) == "item_c"
    
    def test_invalid_index_raises(self):
        """Test that invalid indices raise IndexError."""
        registry = IndexedRegistry({0: "item"})
        
        with pytest.raises(IndexError):
            registry.get(999)
    
    def test_validation_at_init(self):
        """Test that items are validated at registration time."""
        def validator(item):
            return item.startswith("valid_")
        
        # Should succeed
        registry = IndexedRegistry(
            {0: "valid_item"},
            validator=validator
        )
        assert registry.get(0) == "valid_item"
        
        # Should fail
        with pytest.raises(ValueError):
            IndexedRegistry(
                {0: "invalid_item"},
                validator=validator
            )
    
    def test_get_context(self):
        """Test generating LLM-friendly context."""
        registry = IndexedRegistry(
            {0: "item_a", 1: "item_b"},
            describer=lambda x: x.upper()
        )
        
        context = registry.get_context()
        assert context[0] == "ITEM_A"
        assert context[1] == "ITEM_B"
    
    def test_len_and_contains(self):
        """Test __len__ and __contains__."""
        registry = IndexedRegistry({0: "a", 1: "b", 2: "c"})
        
        assert len(registry) == 3
        assert 0 in registry
        assert 1 in registry
        assert 999 not in registry
