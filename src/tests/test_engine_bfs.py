"""Test cases for BFS engine functionality."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, Any, List

from src.tools.engine_bfs import advance_bfs, EDGE_ORDER
from src.tools.session_store import TraverseSession, Frame
from src.tools.token_budget import TokenBudget


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
    def __init__(self, source_uuid: str, target_uuid: str, name: str = "RELATES_TO", fact: str = None):
        self.source_node_uuid = source_uuid
        self.target_node_uuid = target_uuid
        self.name = name
        self.fact = fact or f"{source_uuid} relates to {target_uuid}"
        self.created_at = None
        self.valid_at = None
        self.invalid_at = None
        self.episodes = []


class TestEngineBFS:
    """Test cases for BFS engine."""
    
    @pytest_asyncio.fixture
    async def mock_graphiti(self):
        """Create a mock Graphiti client."""
        mock = MagicMock()
        mock.driver = MagicMock()
        return mock
    
    @pytest_asyncio.fixture
    async def mock_functions(self):
        """Create mock formatting and retrieval functions."""
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
        
        async def get_node_by_uuid(client, uuid):
            # Simple mock that returns FakeNode
            if uuid.startswith("missing"):
                return None
            return FakeNode(uuid)
        
        return {
            "format_node_result": format_node_result,
            "format_edge_for_traverse": format_edge_for_traverse,
            "get_node_by_uuid": get_node_by_uuid,
        }
    
    @pytest.mark.asyncio
    async def test_first_page_returns_root_node_no_edges(self, mock_graphiti, mock_functions):
        """Test that first page returns root node when it has no edges."""
        # Mock EntityEdge.get_by_node_uuid to return no edges initially
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid", 
                   new_callable=AsyncMock) as mock_get_edges:
            mock_get_edges.return_value = []
            
            sess = TraverseSession(
                root_uuid="N1",
                max_depth=2,
                strategy="bfs",
                edge_ordering="uuid",
                query_hash="N1:2",
                frontier=[],
                visited=[],
                yielded_edges=0,
            )
            
            result, has_more, tokens = await advance_bfs(
                sess, mock_graphiti,
                **mock_functions
            )
            
            # Flat structure assertions
            assert result["start"] == "N1"
            assert "N1" in result["nodes"]
            assert result["nodes"]["N1"]["uuid"] == "N1"
            assert result["nodes"]["N1"]["name"] == "Node N1"
            assert sess.visited == ["N1"]
            # First call adds root to frontier, then processes it (finds no edges)
            assert len(sess.frontier) == 0  # Processed and removed
            assert has_more is False
    
    @pytest.mark.asyncio
    async def test_first_page_with_edges(self, mock_graphiti, mock_functions):
        """Test first page returns root node AND its edges (common case)."""
        edges_n1 = [
            FakeEdge("N1", "N2"),
            FakeEdge("N1", "N3"),
            FakeEdge("N1", "N4"),
        ]
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid", 
                   new_callable=AsyncMock) as mock_get_edges:
            def get_edges_for_node(driver, node_uuid):
                if node_uuid == "N1":
                    return edges_n1
                return []
            
            mock_get_edges.side_effect = get_edges_for_node
            
            sess = TraverseSession(
                root_uuid="N1",
                max_depth=2,
                strategy="bfs",
                edge_ordering="uuid",
                query_hash="N1:2",
                frontier=[],
                visited=[],
                yielded_edges=0,
            )
            
            # First page should return root node AND process its edges
            result, has_more, tokens = await advance_bfs(
                sess, mock_graphiti,
                **mock_functions
            )
            
            # Should have root node in flat structure
            assert result["start"] == "N1"
            assert "N1" in result["nodes"]
            assert result["nodes"]["N1"]["uuid"] == "N1"
            
            # Should have processed edges from root (flat structure)
            assert len(result["edges"]) == 3
            assert result["edges"][0]["source"] == "N1"
            assert result["edges"][0]["target"] == "N2"
            assert result["edges"][1]["source"] == "N1"
            assert result["edges"][1]["target"] == "N3"
            assert result["edges"][2]["source"] == "N1"
            assert result["edges"][2]["target"] == "N4"
            
            # All target nodes should be in nodes dict
            assert "N2" in result["nodes"]
            assert "N3" in result["nodes"]
            assert "N4" in result["nodes"]
            
            # All nodes should be visited
            assert "N1" in sess.visited
            assert "N2" in sess.visited
            assert "N3" in sess.visited
            assert "N4" in sess.visited
            
            # Frontier should be empty (child nodes had no edges)
            assert len(sess.frontier) == 0
            assert has_more is False
    
    @pytest.mark.asyncio 
    async def test_first_page_partial_due_to_budget(self, mock_graphiti, mock_functions):
        """Test first page returns root and only some edges due to token budget."""
        # Many edges from root
        edges_n1 = [FakeEdge("N1", f"N{i}") for i in range(2, 12)]  # 10 edges
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid", 
                   new_callable=AsyncMock) as mock_get_edges:
            def get_edges_for_node(driver, node_uuid):
                if node_uuid == "N1":
                    return edges_n1
                return []
            
            mock_get_edges.side_effect = get_edges_for_node
            
            # Small budget to force pagination (but big enough for at least one edge)
            small_budget = TokenBudget(limit=300)
            
            sess = TraverseSession(
                root_uuid="N1",
                max_depth=2,
                strategy="bfs",
                edge_ordering="uuid",
                query_hash="N1:2",
                frontier=[],
                visited=[],
                yielded_edges=0,
            )
            
            result, has_more, tokens = await advance_bfs(
                sess, mock_graphiti,
                budget=small_budget,
                **mock_functions
            )
            
            # Should have root node in flat structure
            assert result["start"] == "N1"
            assert "N1" in result["nodes"]
            assert result["nodes"]["N1"]["uuid"] == "N1"
            
            # Should have some edges but not all (budget limit)
            assert len(result["edges"]) > 0
            assert len(result["edges"]) < 10  # Not all edges
            
            # Should have more pages
            assert has_more is True
            
            # Frame should be back in frontier with partial progress
            assert len(sess.frontier) > 0
            assert sess.frontier[0].node_uuid == "N1"
            assert sess.frontier[0].next_edge_index > 0
    
    @pytest.mark.asyncio
    async def test_depth_zero_returns_only_node(self, mock_graphiti, mock_functions):
        """Test that depth=0 returns only the node without edges."""
        sess = TraverseSession(
            root_uuid="N1",
            max_depth=0,  # No traversal
            strategy="bfs",
            edge_ordering="uuid",
            query_hash="N1:0",
            frontier=[],
            visited=[],
            yielded_edges=0,
        )
        
        result, has_more, tokens = await advance_bfs(
            sess, mock_graphiti,
            **mock_functions
        )
        
        assert result["start"] == "N1"
        assert "N1" in result["nodes"]
        assert result["edges"] == []
        assert has_more is False
        assert len(sess.frontier) == 0  # No frontier for depth=0
    
    @pytest.mark.asyncio
    async def test_missing_node_error_handling(self, mock_graphiti, mock_functions):
        """Test handling of missing nodes."""
        sess = TraverseSession(
            root_uuid="missing-node",
            max_depth=1,
            strategy="bfs",
            edge_ordering="uuid",
            query_hash="missing:1",
            frontier=[],
            visited=[],
            yielded_edges=0,
        )
        
        result, has_more, tokens = await advance_bfs(
            sess, mock_graphiti,
            **mock_functions
        )
        
        assert result["start"] == "missing-node"
        assert "missing-node" in result["nodes"]
        assert result["nodes"]["missing-node"]["error"] == "Node not found"
        assert has_more is False
    
    @pytest.mark.asyncio
    async def test_edge_processing_and_frontier_expansion(self, mock_graphiti, mock_functions):
        """Test that edges are processed and frontier expands correctly."""
        # Setup edges for each node
        edges_n1 = [
            FakeEdge("N1", "N2"),
            FakeEdge("N1", "N3"),
        ]
        edges_n2 = []  # N2 has no outgoing edges
        edges_n3 = []  # N3 has no outgoing edges
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid", 
                   new_callable=AsyncMock) as mock_get_edges:
            # Return appropriate edges for each node
            def get_edges_for_node(driver, node_uuid):
                if node_uuid == "N1":
                    return edges_n1
                elif node_uuid == "N2":
                    return edges_n2
                elif node_uuid == "N3":
                    return edges_n3
                return []
            
            mock_get_edges.side_effect = get_edges_for_node
            
            sess = TraverseSession(
                root_uuid="N1",
                max_depth=2,
                strategy="bfs",
                edge_ordering="uuid",
                query_hash="N1:2",
                frontier=[Frame("N1", 2, 0)],
                visited=["N1"],
                yielded_edges=0,
            )
            
            result, has_more, tokens = await advance_bfs(
                sess, mock_graphiti,
                **mock_functions
            )
            
            # Check edges were added (only from N1) - flat structure
            assert len(result["edges"]) == 2
            assert result["edges"][0]["source"] == "N1"
            assert result["edges"][0]["target"] == "N2"
            assert result["edges"][1]["source"] == "N1"
            assert result["edges"][1]["target"] == "N3"
            
            # Target nodes should be in nodes dict
            assert "N2" in result["nodes"]
            assert "N3" in result["nodes"]
            
            # Check frontier is empty (N2 and N3 were processed but had no edges)
            assert len(sess.frontier) == 0
            assert has_more is False
            
            # Check visited
            assert "N2" in sess.visited
            assert "N3" in sess.visited
    
    @pytest.mark.asyncio
    async def test_cycle_detection(self, mock_graphiti, mock_functions):
        """Test that cycles are detected and marked."""
        # Create edges that form a cycle: N1 -> N2 -> N1
        edges_n1 = [FakeEdge("N1", "N2")]
        edges_n2 = [FakeEdge("N2", "N1")]  # Back to N1 (cycle)
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid", 
                   new_callable=AsyncMock) as mock_get_edges:
            # Return appropriate edges for each node
            def get_edges_for_node(driver, node_uuid):
                if node_uuid == "N1":
                    return edges_n1
                elif node_uuid == "N2":
                    return edges_n2
                return []
            
            mock_get_edges.side_effect = get_edges_for_node
            
            sess = TraverseSession(
                root_uuid="N1",
                max_depth=3,
                strategy="bfs",
                edge_ordering="uuid",
                query_hash="N1:3",
                frontier=[Frame("N1", 3, 0)],
                visited=["N1"],
                yielded_edges=0,
            )
            
            # Single advance processes both N1->N2 and N2->N1 (within token budget)
            result, has_more, _ = await advance_bfs(
                sess, mock_graphiti,
                **mock_functions
            )
            
            # Should have processed both edges
            assert len(result["edges"]) == 2
            
            # First edge: N1->N2 (normal)
            assert result["edges"][0]["source"] == "N1"
            assert result["edges"][0]["target"] == "N2"
            assert "N2" in result["nodes"]
            assert "cyclic_reference" not in result["edges"][0]["target"]
            
            # Second edge: N2->N1 (cyclic reference detected) - flat structure  
            assert result["edges"][1]["source"] == "N2"
            assert result["edges"][1]["target"] == "N1"
            # Note: In flat structure, cycles are handled by not re-adding node to nodes dict
    
    @pytest.mark.asyncio
    async def test_token_budget_interruption(self, mock_graphiti, mock_functions):
        """Test that traversal stops when token budget is exceeded."""
        # Create many edges
        edges = [FakeEdge("N1", f"N{i}") for i in range(2, 10)]
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid", 
                   new_callable=AsyncMock) as mock_get_edges:
            mock_get_edges.return_value = edges
            
            # Small budget that will be exceeded (but big enough for at least one edge)
            small_budget = TokenBudget(limit=250)
            
            sess = TraverseSession(
                root_uuid="N1",
                max_depth=2,
                strategy="bfs",
                edge_ordering="uuid",
                query_hash="N1:2",
                frontier=[Frame("N1", 2, 0)],
                visited=["N1"],
                yielded_edges=0,
            )
            
            result, has_more, tokens = await advance_bfs(
                sess, mock_graphiti,
                budget=small_budget,
                **mock_functions
            )
            
            # Should have stopped before processing all edges
            assert has_more is True
            assert len(result["edges"]) < len(edges)
            
            # Frame should be back in frontier with updated index
            assert len(sess.frontier) > 0
            assert sess.frontier[0].node_uuid == "N1"
            assert sess.frontier[0].next_edge_index > 0
    
    @pytest.mark.asyncio
    async def test_resume_from_middle_of_edges(self, mock_graphiti, mock_functions):
        """Test resuming traversal from middle of edge list."""
        edges_n1 = [FakeEdge("N1", f"N{i}") for i in range(2, 6)]
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid", 
                   new_callable=AsyncMock) as mock_get_edges:
            # Return edges only for N1, empty for other nodes
            def get_edges_for_node(driver, node_uuid):
                if node_uuid == "N1":
                    return edges_n1
                return []
            
            mock_get_edges.side_effect = get_edges_for_node
            
            # Start with frame that's already partially processed
            sess = TraverseSession(
                root_uuid="N1",
                max_depth=2,
                strategy="bfs",
                edge_ordering="uuid",
                query_hash="N1:2",
                frontier=[Frame("N1", 2, 2)],  # Start from edge index 2
                visited=["N1", "N2", "N3"],  # Already visited N2, N3
                yielded_edges=2,
            )
            
            result, has_more, tokens = await advance_bfs(
                sess, mock_graphiti,
                **mock_functions
            )
            
            # Should process edges starting from index 2 (N4 and N5)
            assert len(result["edges"]) == 2
            
            # N4 and N5 should be new (not cyclic) - flat structure
            assert result["edges"][0]["source"] == "N1"
            assert result["edges"][0]["target"] == "N4"
            assert "N4" in result["nodes"]
            
            assert result["edges"][1]["source"] == "N1"
            assert result["edges"][1]["target"] == "N5"
            assert "N5" in result["nodes"]
            
            # Frontier should be empty (N4, N5 processed but have no edges)
            assert len(sess.frontier) == 0
            assert has_more is False
    
    @pytest.mark.asyncio
    async def test_edge_ordering_stability(self, mock_graphiti, mock_functions):
        """Test that edge ordering is stable across pages."""
        edges = [
            FakeEdge("N1", "N2", "TYPE_B"),
            FakeEdge("N1", "N3", "TYPE_A"),
            FakeEdge("N1", "N4", "TYPE_B"),
        ]
        
        with patch("src.tools.engine_bfs.EntityEdge.get_by_node_uuid", 
                   new_callable=AsyncMock) as mock_get_edges:
            mock_get_edges.return_value = edges
            
            sess = TraverseSession(
                root_uuid="N1",
                max_depth=1,
                strategy="bfs",
                edge_ordering="type_then_uuid",  # Should sort by type first
                query_hash="N1:1",
                frontier=[Frame("N1", 1, 0)],
                visited=["N1"],
                yielded_edges=0,
            )
            
            result, has_more, tokens = await advance_bfs(
                sess, mock_graphiti,
                **mock_functions
            )
            
            # Check edges are ordered by type (TYPE_A before TYPE_B)
            assert len(result["edges"]) == 3
            edge_types = [e["type"] for e in result["edges"]]
            assert edge_types == ["TYPE_A", "TYPE_B", "TYPE_B"]
    
    @pytest.mark.asyncio
    async def test_empty_frontier_returns_complete(self, mock_graphiti, mock_functions):
        """Test that empty frontier means traversal is complete."""
        sess = TraverseSession(
            root_uuid="N1",
            max_depth=1,
            strategy="bfs",
            edge_ordering="uuid",
            query_hash="N1:1",
            frontier=[],  # Empty frontier
            visited=["N1", "N2", "N3"],
            yielded_edges=5,
        )
        
        result, has_more, tokens = await advance_bfs(
            sess, mock_graphiti,
            **mock_functions
        )
        
        assert result["edges"] == []
        assert has_more is False