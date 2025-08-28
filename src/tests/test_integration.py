"""Integration tests for cursor-based pagination."""

import pytest
import pytest_asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.tools.traverse_wrapper import (
    traverse_knowledge_graph_paginated,
)
from src.tools.traverse_knowledge_graph import (
    traverse_knowledge_graph_impl,
    get_node_by_uuid,
)
from src.tools.session_store import SessionStore


class FakeNode:
    """Fake node for testing."""
    def __init__(self, uuid: str, name: str = None):
        self.uuid = uuid
        self.name = name or f"Node {uuid}"
        self.summary = f"Summary of {name or uuid}"
        self.labels = ["Entity"]
        self.group_id = "test-group"
        self.created_at = None
        self.attributes = {}


class FakeEdge:
    """Fake edge for testing."""
    def __init__(self, source: str, target: str, name: str = "RELATES_TO"):
        self.source_node_uuid = source
        self.target_node_uuid = target
        self.name = name
        self.fact = f"{source} {name} {target}"
        self.created_at = None
        self.valid_at = None
        self.invalid_at = None
        self.episodes = []


def create_test_graph():
    """Create a test graph structure.
    
    Returns a dictionary mapping node UUIDs to their edges.
    
    Graph structure:
        N1
        ├── N2
        │   ├── N4
        │   └── N5
        └── N3
            └── N6
                └── N7
    """
    return {
        "N1": [FakeEdge("N1", "N2"), FakeEdge("N1", "N3")],
        "N2": [FakeEdge("N2", "N4"), FakeEdge("N2", "N5")],
        "N3": [FakeEdge("N3", "N6")],
        "N4": [],
        "N5": [],
        "N6": [FakeEdge("N6", "N7")],
        "N7": [],
    }


def collect_all_edges(result: Dict[str, Any]) -> List[str]:
    """Collect all edge targets from a flat structure result."""
    edges = []
    
    # For flat structure, just collect targets from edges array
    if "edges" in result:
        for edge in result["edges"]:
            target_uuid = edge.get("target")
            if target_uuid:
                edges.append(target_uuid)
    
    return sorted(edges)


class TestIntegration:
    """Integration tests for pagination vs non-pagination."""
    
    @pytest_asyncio.fixture
    async def mock_graphiti(self):
        """Create a mock Graphiti client."""
        mock = MagicMock()
        mock.driver = MagicMock()
        return mock
    
    @pytest_asyncio.fixture
    async def setup_graph(self):
        """Setup graph data and mocks."""
        graph = create_test_graph()
        
        async def get_node_by_uuid(client, uuid):
            if uuid in graph:
                return FakeNode(uuid)
            return None
        
        def format_node_result(node):
            if node is None:
                return None
            return {
                "uuid": node.uuid,
                "name": node.name,
                "summary": node.summary,
            }
        
        def format_edge_for_traverse(edge, target_data):
            return {
                "type": edge.name,
                "fact": edge.fact,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "target": target_data,
            }
        
        return {
            "graph": graph,
            "get_node_by_uuid": get_node_by_uuid,
            "format_node_result": format_node_result,
            "format_edge_for_traverse": format_edge_for_traverse,
        }
    
    @pytest.mark.asyncio
    async def test_single_page_matches_legacy(self, mock_graphiti, setup_graph):
        """Test that single-page result matches legacy implementation."""
        graph_data = setup_graph["graph"]
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid") as mock_get_edges:
            # Get the actual module from sys.modules to patch it correctly
            actual_module = sys.modules['src.tools.traverse_knowledge_graph']
            with patch.object(actual_module, 'get_node_by_uuid', new_callable=AsyncMock) as mock_get_node:
                # Setup edge mocks
                async def get_edges_for_node(driver, node_uuid):
                    return graph_data.get(node_uuid, [])
                
                mock_get_edges.side_effect = get_edges_for_node
                mock_get_node.side_effect = setup_graph["get_node_by_uuid"]
                
                # Get legacy result
                legacy_result = await traverse_knowledge_graph_impl(
                    mock_graphiti,
                    "N1",
                    depth=2,
                )
                
                # Get paginated result (should be single page for depth=2)
                paginated_result = await traverse_knowledge_graph_paginated(
                    mock_graphiti,
                    start_node_uuid="N1",
                    depth=2,
                    format_node_result=setup_graph["format_node_result"],
                    format_edge_for_traverse=setup_graph["format_edge_for_traverse"],
                    get_node_by_uuid=setup_graph["get_node_by_uuid"],
                )
                
                # Compare flat structure - both should have same start
                assert legacy_result["start"] == paginated_result["start"]
                assert legacy_result["start"] == "N1"
                
                # Compare edge targets
                legacy_edges = collect_all_edges(legacy_result)
                paginated_edges = collect_all_edges(paginated_result)
                assert legacy_edges == paginated_edges
    
    @pytest.mark.asyncio
    async def test_multi_page_complete_traversal(self, mock_graphiti, setup_graph):
        """Test that multi-page traversal eventually returns all nodes."""
        graph_data = setup_graph["graph"]
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid") as mock_get_edges:
            # Setup edge mock
            async def get_edges_for_node(driver, node_uuid):
                return graph_data.get(node_uuid, [])
            
            mock_get_edges.side_effect = get_edges_for_node
            
            # Use very small token budget to force pagination
            with patch("src.tools.traverse_wrapper.TokenBudget") as MockBudget:
                # Create a budget factory that returns new instance each time
                class SmallBudget:
                    def __init__(self, limit=20000):
                        self.edge_count = 0
                        self.limit = limit  # Add limit attribute
                        self.max_tokens = limit  # Add max_tokens alias
                    
                    def can_add_edge(self, result, edge):
                        # Allow root node and 1 edge per page
                        if self.edge_count >= 1:
                            return False
                        self.edge_count += 1
                        return True
                
                # Use side_effect to create new instance each time
                MockBudget.side_effect = lambda: SmallBudget()
                
                # Collect all nodes through pagination
                all_nodes = set()
                cursor_token = None
                page_count = 0
                max_pages = 20  # Safety limit
                
                while page_count < max_pages:
                    result = await traverse_knowledge_graph_paginated(
                        mock_graphiti,
                        start_node_uuid="N1" if cursor_token is None else None,
                        depth=3,
                        cursor_token=cursor_token,
                        format_node_result=setup_graph["format_node_result"],
                format_edge_for_traverse=setup_graph["format_edge_for_traverse"],
                get_node_by_uuid=setup_graph["get_node_by_uuid"],
                    )
                    
                    # Collect nodes from this page (flat structure)
                    # Add start node
                    if result.get("start"):
                        all_nodes.add(result["start"])
                    
                    # Add all nodes from the nodes dictionary
                    if result.get("nodes"):
                        all_nodes.update(result["nodes"].keys())
                    
                    # Also collect target UUIDs from edges
                    for edge in result.get("edges", []):
                        target_uuid = edge.get("target")  # In flat structure, target is just a UUID string
                        if target_uuid:
                            all_nodes.add(target_uuid)
                    
                    page_count += 1
                    
                    # Check for more pages
                    if not result.get("cursor", {}).get("has_more", False):
                        break
                    
                    cursor_token = result["cursor"]["token"]
                
                # Should have visited all reachable nodes
                assert "N1" in all_nodes
                assert "N2" in all_nodes
                assert "N3" in all_nodes
                assert "N6" in all_nodes  # Through N3
                assert "N7" in all_nodes  # Through N6 at depth 3
    
    @pytest.mark.asyncio
    async def test_cursor_expiry_handling(self, mock_graphiti, setup_graph):
        """Test handling of expired cursors."""
        from src.tools.session_store import CursorExpired
        
        # Get first page with small token budget to force pagination
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid") as mock_get_edges:
            mock_get_edges.return_value = [FakeEdge("N1", "N2")]
            
            # Use actual TokenBudget with small limit to force pagination
            with patch("src.tools.traverse_wrapper.TokenBudget") as MockBudget:
                from src.tools.token_budget import TokenBudget
                MockBudget.side_effect = lambda: TokenBudget(limit=100)  # Very small limit
                
                result = await traverse_knowledge_graph_paginated(
                    mock_graphiti,
                    start_node_uuid="N1",
                    depth=2,
                    format_node_result=setup_graph["format_node_result"],
                    format_edge_for_traverse=setup_graph["format_edge_for_traverse"],
                    get_node_by_uuid=setup_graph["get_node_by_uuid"],
                )
                
                # Should have pagination with small budget
                assert result.get("cursor", {}).get("has_more", False)
            
            if result["cursor"]["has_more"]:
                cursor_token = result["cursor"]["token"]
                
                # Simulate token expiry
                with patch("src.tools.traverse_wrapper._session_store.verify_token") as mock_verify:
                    mock_verify.side_effect = CursorExpired("Token expired")
                    
                    with pytest.raises(CursorExpired):
                        await traverse_knowledge_graph_paginated(
                            mock_graphiti,
                            cursor_token=cursor_token,
                            format_node_result=setup_graph["format_node_result"],
                format_edge_for_traverse=setup_graph["format_edge_for_traverse"],
                get_node_by_uuid=setup_graph["get_node_by_uuid"],
                        )
    
    @pytest.mark.asyncio
    async def test_depth_zero_no_pagination(self, mock_graphiti, setup_graph):
        """Test that depth=0 returns only root without pagination."""
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid") as mock_get_edges:
            mock_get_edges.return_value = []
            
            result = await traverse_knowledge_graph_paginated(
                mock_graphiti,
                start_node_uuid="N1",
                depth=0,
                format_node_result=setup_graph["format_node_result"],
                format_edge_for_traverse=setup_graph["format_edge_for_traverse"],
                get_node_by_uuid=setup_graph["get_node_by_uuid"],
            )
            
            # Should have root node in flat structure
            assert result["start"] == "N1"
            assert "N1" in result["nodes"]
            assert result["nodes"]["N1"]["name"] == "Node N1"
            
            # Should have no edges
            assert len(result.get("edges", [])) == 0
            
            # Should not need pagination
            assert result["cursor"]["has_more"] is False
    
    @pytest.mark.asyncio
    async def test_cycle_handling_consistency(self, mock_graphiti, setup_graph):
        """Test that cycles are handled consistently between implementations."""
        # Create a graph with cycle: N1 -> N2 -> N3 -> N1
        cyclic_graph = {
            "N1": [FakeEdge("N1", "N2")],
            "N2": [FakeEdge("N2", "N3")],
            "N3": [FakeEdge("N3", "N1")],  # Cycle back to N1
        }
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid") as mock_get_edges:
            with patch("graphiti_core.edges.EntityEdge.get_by_node_uuid") as mock_legacy_edges:
                # Setup edge mocks
                async def get_edges_for_node(driver, node_uuid):
                    return cyclic_graph.get(node_uuid, [])
                
                mock_get_edges.side_effect = get_edges_for_node
                mock_legacy_edges.side_effect = get_edges_for_node
                
                # Get legacy result
                legacy_result = await traverse_knowledge_graph_impl(
                    mock_graphiti,
                    "N1",
                    depth=4,  # Deep enough to encounter cycle
                )
                
                # Get paginated result
                paginated_result = await traverse_knowledge_graph_paginated(
                    mock_graphiti,
                    start_node_uuid="N1",
                    depth=4,
                    format_node_result=setup_graph["format_node_result"],
                format_edge_for_traverse=setup_graph["format_edge_for_traverse"],
                get_node_by_uuid=setup_graph["get_node_by_uuid"],
                )
                
                # Count cyclic references in flat structure
                def count_cycles(result):
                    # In flat structure, cycles are detected by checking if
                    # an edge points to an already visited node
                    visited_nodes = set()
                    cycle_count = 0
                    
                    # Add start node to visited
                    if "start" in result:
                        visited_nodes.add(result["start"])
                    
                    # Check edges for cycles
                    for edge in result.get("edges", []):
                        target = edge.get("target")
                        if target and target in visited_nodes:
                            cycle_count += 1
                        elif target:
                            visited_nodes.add(target)
                    
                    return cycle_count
                
                legacy_cycles = count_cycles(legacy_result)
                paginated_cycles = count_cycles(paginated_result)
                
                # Both should detect the cycle
                assert legacy_cycles > 0
                assert paginated_cycles > 0
    
    @pytest.mark.asyncio
    async def test_extended_function_mode_selection(self, mock_graphiti, setup_graph):
        """Test that extended function selects appropriate mode."""
        graph_data = setup_graph["graph"]
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid") as mock_get_edges:
            with patch("graphiti_core.edges.EntityEdge.get_by_node_uuid") as mock_legacy_edges:
                async def get_edges_for_node(driver, node_uuid):
                    return graph_data.get(node_uuid, [])
                
                mock_get_edges.side_effect = get_edges_for_node
                mock_legacy_edges.side_effect = get_edges_for_node
                
                # Both shallow and deep depth now use pagination via traverse_knowledge_graph_impl
                shallow_result = await traverse_knowledge_graph_impl(
                    mock_graphiti,
                    start_node_uuid="N1",
                    depth=1,
                )
                
                # Should have flat structure
                assert "start" in shallow_result
                assert "nodes" in shallow_result
                assert "edges" in shallow_result
                
                # Deep depth also uses pagination
                deep_result = await traverse_knowledge_graph_impl(
                    mock_graphiti,
                    start_node_uuid="N1",
                    depth=3,
                )
                
                # Should also have flat structure
                assert "start" in deep_result
                assert "nodes" in deep_result