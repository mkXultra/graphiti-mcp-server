"""Integration tests using real database for cursor-based pagination."""

import pytest
import pytest_asyncio
import os
from typing import Dict, Any, List, Set

from graphiti_core import Graphiti
from src.tools.traverse_wrapper import (
    traverse_knowledge_graph_paginated,
)
from src.tools.traverse_knowledge_graph import (
    traverse_knowledge_graph_impl,
    format_node_result,
    format_edge_for_traverse,
    get_node_by_uuid,
)
from src.tools.token_budget import TokenBudget

# Test database configuration
TEST_NEO4J_URI = os.getenv('TEST_NEO4J_URI', 'bolt://localhost:7688')
TEST_NEO4J_USER = os.getenv('TEST_NEO4J_USER', 'neo4j')
TEST_NEO4J_PASSWORD = os.getenv('TEST_NEO4J_PASSWORD', 'testpassword')

# Known test data from the test database
# Bob Johnson is connected to projects and other entities
BOB_JOHNSON_UUID = "3e6968a4-2288-4681-8f45-e6f97ac94869"


def collect_all_node_uuids(result: Dict[str, Any]) -> Set[str]:
    """Collect all node UUIDs from a flat structure traversal result."""
    uuids = set()
    
    # For flat structure, just get all UUIDs from nodes dict
    if "nodes" in result and isinstance(result["nodes"], dict):
        uuids.update(result["nodes"].keys())
    
    # Also add start UUID if present
    if "start" in result:
        uuids.add(result["start"])
    
    return uuids


class TestIntegrationWithRealDB:
    """Integration tests using real Neo4j test database."""
    
    @pytest_asyncio.fixture
    async def graphiti_client(self):
        """Create a real Graphiti client connected to test database."""
        client = Graphiti(
            uri=TEST_NEO4J_URI,
            user=TEST_NEO4J_USER,
            password=TEST_NEO4J_PASSWORD,
        )
        yield client
        await client.driver.close()
    
    @pytest.mark.asyncio
    async def test_single_page_matches_legacy(self, graphiti_client):
        """Test that single-page result matches legacy implementation."""
        # Use shallow depth to ensure single page
        depth = 1
        
        # Get legacy result
        legacy_result = await traverse_knowledge_graph_impl(
            graphiti_client,
            BOB_JOHNSON_UUID,
            depth=depth,
        )
        
        # Get paginated result (should be single page for depth=1)
        paginated_result = await traverse_knowledge_graph_paginated(
            graphiti_client,
            start_node_uuid=BOB_JOHNSON_UUID,
            depth=depth,
            format_node_result=format_node_result,
            format_edge_for_traverse=format_edge_for_traverse,
            get_node_by_uuid=get_node_by_uuid,
        )
        
        # Compare root nodes
        # Both should have same start UUID in flat structure
        assert legacy_result["start"] == paginated_result["start"]
        # Compare nodes in flat structure
        assert BOB_JOHNSON_UUID in legacy_result["nodes"]
        assert BOB_JOHNSON_UUID in paginated_result["nodes"]
        assert legacy_result["nodes"][BOB_JOHNSON_UUID]["name"] == paginated_result["nodes"][BOB_JOHNSON_UUID]["name"]
        
        # Compare collected UUIDs
        legacy_uuids = collect_all_node_uuids(legacy_result)
        paginated_uuids = collect_all_node_uuids(paginated_result)
        
        # Should have the same nodes
        assert legacy_uuids == paginated_uuids
        
        # Should not need pagination for depth=1
        assert paginated_result["cursor"]["has_more"] is False
    
    @pytest.mark.asyncio
    async def test_multi_page_complete_traversal(self, graphiti_client):
        """Test that multi-page traversal eventually returns all nodes."""
        # Use very small token budget to force pagination
        from unittest.mock import patch
        from src.tools.token_budget import TokenBudget
        
        # Patch TokenBudget to use a very small limit
        with patch("src.tools.traverse_wrapper.TokenBudget") as MockBudget:
            # Use real TokenBudget with small limit to force pagination
            MockBudget.side_effect = lambda: TokenBudget(limit=200)  # Very small limit
            
            # Collect all nodes through pagination
            all_uuids = set()
            cursor_token = None
            page_count = 0
            max_pages = 20  # Safety limit
            
            depth = 2
            
            while page_count < max_pages:
                result = await traverse_knowledge_graph_paginated(
                    graphiti_client,
                    start_node_uuid=BOB_JOHNSON_UUID if cursor_token is None else None,
                    depth=depth,
                    cursor_token=cursor_token,
                    format_node_result=format_node_result,
                    format_edge_for_traverse=format_edge_for_traverse,
                    get_node_by_uuid=get_node_by_uuid,
                )
                
                # Collect UUIDs from this page
                page_uuids = collect_all_node_uuids(result)
                all_uuids.update(page_uuids)
                
                page_count += 1
                
                # Check for more pages
                if not result.get("cursor", {}).get("has_more", False):
                    break
                
                cursor_token = result["cursor"]["token"]
            
            # Get the full result without pagination for comparison
            full_result = await traverse_knowledge_graph_impl(
                graphiti_client,
                BOB_JOHNSON_UUID,
                depth=depth,
            )
            full_uuids = collect_all_node_uuids(full_result)
            
            # Should have collected all the same nodes
            assert all_uuids == full_uuids
            
            # Should have needed multiple pages
            assert page_count > 1
    
    @pytest.mark.asyncio
    async def test_depth_zero_no_pagination(self, graphiti_client):
        """Test that depth=0 returns only root without pagination."""
        result = await traverse_knowledge_graph_paginated(
            graphiti_client,
            start_node_uuid=BOB_JOHNSON_UUID,
            depth=0,
            format_node_result=format_node_result,
            format_edge_for_traverse=format_edge_for_traverse,
            get_node_by_uuid=get_node_by_uuid,
        )
        
        # Should have root node
        assert result["start"] == BOB_JOHNSON_UUID
        assert BOB_JOHNSON_UUID in result["nodes"]
        assert result["nodes"][BOB_JOHNSON_UUID]["name"] == "Bob Johnson"
        
        # Should have no edges
        assert len(result.get("edges", [])) == 0
        
        # Should not need pagination
        assert result["cursor"]["has_more"] is False
    
    @pytest.mark.asyncio
    async def test_impl_function_with_real_data(self, graphiti_client):
        """Test traverse_knowledge_graph_impl with real database data."""
        # Test shallow depth
        shallow_result = await traverse_knowledge_graph_impl(
            graphiti_client,
            start_node_uuid=BOB_JOHNSON_UUID,
            depth=1,
        )
        
        # Should return valid flat structure result
        assert shallow_result["start"] == BOB_JOHNSON_UUID
        assert BOB_JOHNSON_UUID in shallow_result["nodes"]
        assert "edges" in shallow_result
        
        # Test deeper depth (may trigger pagination)
        deep_result = await traverse_knowledge_graph_impl(
            graphiti_client,
            start_node_uuid=BOB_JOHNSON_UUID,
            depth=3,
        )
        
        # Should return valid flat structure result
        assert deep_result["start"] == BOB_JOHNSON_UUID
        assert BOB_JOHNSON_UUID in deep_result["nodes"]
        assert "edges" in deep_result
    
    @pytest.mark.asyncio
    async def test_cursor_expiry_handling(self, graphiti_client):
        """Test handling of expired cursors."""
        from src.tools.session_store import CursorExpired
        from unittest.mock import patch
        
        # Start a traversal to get a cursor
        result = await traverse_knowledge_graph_paginated(
            graphiti_client,
            start_node_uuid=BOB_JOHNSON_UUID,
            depth=2,
            format_node_result=format_node_result,
            format_edge_for_traverse=format_edge_for_traverse,
            get_node_by_uuid=get_node_by_uuid,
        )
        
        # If we got a cursor
        if result.get("cursor", {}).get("has_more", False):
            cursor_token = result["cursor"]["token"]
            
            # Simulate token expiry
            with patch("src.tools.traverse_wrapper._session_store.verify_token") as mock_verify:
                mock_verify.side_effect = CursorExpired("Token expired")
                
                with pytest.raises(CursorExpired):
                    await traverse_knowledge_graph_paginated(
                        graphiti_client,
                        cursor_token=cursor_token,
                        format_node_result=format_node_result,
                        format_edge_for_traverse=format_edge_for_traverse,
                        get_node_by_uuid=get_node_by_uuid,
                    )
    
    @pytest.mark.asyncio
    async def test_consistent_ordering_across_pages(self, graphiti_client):
        """Test that edge ordering is consistent across pages."""
        from unittest.mock import patch
        
        # Force small pages to test ordering
        from src.tools.token_budget import TokenBudget
        
        with patch("src.tools.traverse_wrapper.TokenBudget") as MockBudget:
            # Use real TokenBudget with very small limit to force many small pages
            MockBudget.return_value = TokenBudget(limit=150)  # Extremely small limit
            
            # Collect edges in order
            all_edges = []
            cursor_token = None
            
            for _ in range(5):  # Get up to 5 pages
                result = await traverse_knowledge_graph_paginated(
                    graphiti_client,
                    start_node_uuid=BOB_JOHNSON_UUID if cursor_token is None else None,
                    depth=1,
                    cursor_token=cursor_token,
                    format_node_result=format_node_result,
                    format_edge_for_traverse=format_edge_for_traverse,
                    get_node_by_uuid=get_node_by_uuid,
                )
                
                # Collect edge info
                for edge in result.get("edges", []):
                    edge_info = (
                        edge.get("source_node_uuid"),
                        edge.get("target_node_uuid"),
                        edge.get("type"),
                    )
                    all_edges.append(edge_info)
                
                if not result.get("cursor", {}).get("has_more", False):
                    break
                
                cursor_token = result["cursor"]["token"]
            
            # Edges should be consistently ordered
            # (We can't check exact order without knowing the data,
            # but we can verify no duplicates)
            assert len(all_edges) == len(set(all_edges)), "Found duplicate edges across pages"