"""Token budget management for response size control."""

import json
from typing import Any, Dict, List, Union

# Maximum tokens for response (80% of MCP's 25,000 limit for safety)
MAX_RESPONSE_TOKENS = 20_000

# Try to use tiktoken for accurate token counting
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    
    def estimate_tokens(obj: Any) -> int:
        """Estimate token count using tiktoken."""
        if isinstance(obj, str):
            return len(_ENCODER.encode(obj))
        else:
            # Convert to JSON string and count
            json_str = json.dumps(obj, ensure_ascii=False)
            return len(_ENCODER.encode(json_str))
            
except ImportError:
    # Fallback: rough estimation (4 characters ≈ 1 token)
    def estimate_tokens(obj: Any) -> int:
        """Fallback token estimation without tiktoken."""
        if isinstance(obj, str):
            return max(1, len(obj) // 4)
        else:
            json_str = json.dumps(obj, ensure_ascii=False)
            return max(1, len(json_str) // 4)


class TokenBudget:
    """Manages token budget for response size control."""
    
    def __init__(self, limit: int = MAX_RESPONSE_TOKENS):
        """Initialize token budget.
        
        Args:
            limit: Maximum number of tokens allowed
        """
        self.limit = limit
        self.max_tokens = limit  # Alias for compatibility with spec
        self.used = 0
        self._current_state: Any = None
    
    def can_add(self, obj: Any) -> bool:
        """Check if object can be added without exceeding budget.
        
        Args:
            obj: Object to check
            
        Returns:
            True if object can be added, False otherwise
        """
        tokens = estimate_tokens(obj)
        return (self.used + tokens) <= self.limit
    
    def add(self, obj: Any) -> None:
        """Add object to budget tracking.
        
        Args:
            obj: Object being added
        """
        tokens = estimate_tokens(obj)
        self.used += tokens
        self._current_state = obj
    
    def remaining(self) -> int:
        """Get remaining tokens in budget.
        
        Returns:
            Number of tokens remaining
        """
        return self.limit - self.used
    
    def reset(self) -> None:
        """Reset the budget to empty."""
        self.used = 0
        self._current_state = None
    
    def set_current_state(self, state: Any) -> None:
        """Set the current state and update used tokens.
        
        Args:
            state: Current state object
        """
        self._current_state = state
        self.used = estimate_tokens(state)
    
    def can_add_edge(self, result: Dict[str, Any], edge: Dict[str, Any]) -> bool:
        """Check if an edge can be added to the result without exceeding budget.
        
        This is optimized for graph traversal where we're adding edges to a result.
        
        Args:
            result: Current result object with edges list
            edge: Edge object to potentially add
            
        Returns:
            True if edge can be added, False otherwise
        """
        # Estimate current size
        current_tokens = estimate_tokens(result)
        
        # Estimate size with new edge
        # We'll temporarily add the edge to check
        result_copy = json.loads(json.dumps(result))  # Deep copy
        result_copy["edges"].append(edge)
        new_tokens = estimate_tokens(result_copy)
        
        # Check if adding the edge would exceed limit
        return new_tokens <= self.limit