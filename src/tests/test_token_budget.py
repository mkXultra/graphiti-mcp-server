"""Test cases for TokenBudget functionality."""

import pytest
import json
from typing import Dict, Any, List

from src.tools.token_budget import (
    TokenBudget,
    MAX_RESPONSE_TOKENS,
    estimate_tokens,
)


class TestTokenBudget:
    """Test cases for token budget tracking."""
    
    def test_max_response_tokens_default(self):
        """Test that default max tokens is 20,000."""
        assert MAX_RESPONSE_TOKENS == 20_000
    
    def test_estimate_tokens_simple_string(self):
        """Test token estimation for simple strings."""
        # Simple ASCII text
        text = "Hello world"
        tokens = estimate_tokens(text)
        # Rough estimate: ~2-3 tokens for "Hello world"
        assert 2 <= tokens <= 4
        
        # Longer text
        long_text = "The quick brown fox jumps over the lazy dog" * 10
        tokens = estimate_tokens(long_text)
        # Should be proportional to length
        assert tokens > 20
    
    def test_estimate_tokens_json_object(self):
        """Test token estimation for JSON objects."""
        obj = {
            "node": {"uuid": "N1", "name": "Node 1"},
            "edges": [
                {"uuid": "E1", "type": "RELATES_TO"},
                {"uuid": "E2", "type": "RELATES_TO"}
            ]
        }
        tokens = estimate_tokens(obj)
        # JSON structure should have reasonable token count
        assert tokens > 10
    
    def test_estimate_tokens_unicode(self):
        """Test token estimation handles Unicode correctly."""
        unicode_text = "Hello 世界 🌍"
        tokens = estimate_tokens(unicode_text)
        # Should handle unicode without crashing
        assert tokens > 0
    
    def test_token_budget_initialization(self):
        """Test TokenBudget initialization."""
        # Default budget
        budget = TokenBudget()
        assert budget.limit == MAX_RESPONSE_TOKENS
        assert budget.used == 0
        
        # Custom budget
        custom_budget = TokenBudget(limit=1000)
        assert custom_budget.limit == 1000
        assert custom_budget.used == 0
    
    def test_token_budget_can_add_simple(self):
        """Test checking if an object can be added to budget."""
        budget = TokenBudget(limit=100)
        
        small_obj = {"small": "data"}
        assert budget.can_add(small_obj) is True
        
        # Actually add it
        budget.add(small_obj)
        assert budget.used > 0
        assert budget.remaining() < 100
    
    def test_token_budget_cannot_exceed_limit(self):
        """Test that budget prevents exceeding limit."""
        budget = TokenBudget(limit=50)
        
        # Create an object that's too large
        large_obj = {"data": "x" * 1000}  # Very long string
        
        # Should not be able to add
        assert budget.can_add(large_obj) is False
    
    def test_token_budget_accumulation(self):
        """Test that token usage accumulates correctly."""
        budget = TokenBudget(limit=1000)
        
        obj1 = {"id": 1, "data": "first"}
        obj2 = {"id": 2, "data": "second"}
        obj3 = {"id": 3, "data": "third"}
        
        # Add objects sequentially
        assert budget.can_add(obj1) is True
        budget.add(obj1)
        initial_used = budget.used
        
        assert budget.can_add(obj2) is True
        budget.add(obj2)
        assert budget.used > initial_used
        
        assert budget.can_add(obj3) is True
        budget.add(obj3)
        assert budget.used > initial_used
    
    def test_token_budget_remaining(self):
        """Test remaining token calculation."""
        budget = TokenBudget(limit=1000)
        
        assert budget.remaining() == 1000
        
        obj = {"test": "data"}
        budget.add(obj)
        
        assert budget.remaining() < 1000
        assert budget.remaining() == budget.limit - budget.used
    
    def test_token_budget_with_edge_addition(self):
        """Test budget checking for graph traversal edge addition."""
        budget = TokenBudget(limit=500)
        
        # Simulate a result object with edges
        result = {
            "node": {"uuid": "N1", "name": "Root"},
            "edges": []
        }
        
        # Try adding edges one by one
        edge1 = {
            "uuid": "E1",
            "type": "RELATES_TO",
            "target": {
                "node": {"uuid": "N2", "name": "Target1"},
                "edges": []
            }
        }
        
        # Should be able to add first edge
        assert budget.can_add_edge(result, edge1) is True
        result["edges"].append(edge1)
        budget.set_current_state(result)
        
        # Add more edges
        edge2 = {
            "uuid": "E2",
            "type": "RELATES_TO",
            "target": {
                "node": {"uuid": "N3", "name": "Target2"},
                "edges": []
            }
        }
        
        assert budget.can_add_edge(result, edge2) is True
    
    def test_token_budget_prevents_overflow_with_edges(self):
        """Test that budget prevents adding edges that would exceed limit."""
        budget = TokenBudget(limit=100)  # Very small limit
        
        result = {
            "node": {"uuid": "N1"},
            "edges": []
        }
        
        # Create a large edge that would exceed budget
        large_edge = {
            "uuid": "E1",
            "type": "RELATES_TO",
            "fact": "x" * 500,  # Long fact text
            "target": {
                "node": {"uuid": "N2", "description": "y" * 500},
                "edges": []
            }
        }
        
        # Should not be able to add
        assert budget.can_add_edge(result, large_edge) is False
    
    def test_token_budget_reset(self):
        """Test resetting the budget."""
        budget = TokenBudget(limit=1000)
        
        # Add some objects
        budget.add({"data": "test"})
        assert budget.used > 0
        
        # Reset
        budget.reset()
        assert budget.used == 0
        assert budget.remaining() == 1000
    
    def test_token_budget_with_nested_structure(self):
        """Test token counting with deeply nested structures."""
        nested = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": ["item1", "item2", "item3"],
                        "metadata": {
                            "count": 3,
                            "type": "nested"
                        }
                    }
                }
            }
        }
        
        budget = TokenBudget(limit=1000)
        assert budget.can_add(nested) is True
        budget.add(nested)
        assert budget.used > 0
    
    def test_fallback_estimation(self):
        """Test fallback token estimation when tiktoken is not available."""
        # Test the fallback estimation directly
        test_obj = {"key": "value", "number": 42}
        json_str = json.dumps(test_obj, ensure_ascii=False)
        
        # Fallback should be len(json_str) // 4
        fallback_estimate = len(json_str) // 4
        
        # Since we might have tiktoken, compare with reasonable range
        actual = estimate_tokens(test_obj)
        assert actual > 0
        # Should be in reasonable range of fallback
        assert 0.5 * fallback_estimate <= actual <= 2 * fallback_estimate