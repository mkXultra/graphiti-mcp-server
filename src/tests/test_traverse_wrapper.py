"""Test cases for traverse_wrapper with cursor-based pagination."""

import pytest
import pytest_asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.tools.traverse_wrapper import (
    traverse_knowledge_graph_paginated,
)
from src.tools.session_store import (
    CursorExpired,
    InvalidCursor,
    SessionNotFound,
    QueryMismatch,
)


class FakeNode:
    """Fake node for testing."""
    def __init__(self, uuid: str, name: str = None):
        self.uuid = uuid
        self.name = name or f"Node {uuid}"
        self.summary = f"Summary of {name or uuid}"


class FakeEdge:
    """Fake edge for testing."""
    def __init__(self, source: str, target: str):
        self.source_node_uuid = source
        self.target_node_uuid = target
        self.name = "RELATES_TO"
        self.fact = f"{source} relates to {target}"


class TestTraverseWrapper:
    """Test cases for cursor-based traverse wrapper."""
    
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
            if uuid.startswith("missing"):
                return None
            return FakeNode(uuid)
        
        return {
            "format_node_result": format_node_result,
            "format_edge_for_traverse": format_edge_for_traverse,
            "get_node_by_uuid": get_node_by_uuid,
        }
    
    @pytest.mark.asyncio
    async def test_initial_call_requires_start_node(self, mock_graphiti, mock_functions):
        """Test that initial call requires start_node_uuid."""
        with pytest.raises(ValueError, match="start_node_uuid is required"):
            await traverse_knowledge_graph_paginated(
                mock_graphiti,
                start_node_uuid=None,  # Missing
                depth=2,
                cursor_token=None,
                **mock_functions
            )
    
    @pytest.mark.asyncio
    async def test_depth_validation(self, mock_graphiti, mock_functions):
        """Test depth parameter validation."""
        # Negative depth
        with pytest.raises(ValueError, match="depth must be between 0 and 5"):
            await traverse_knowledge_graph_paginated(
                mock_graphiti,
                start_node_uuid="N1",
                depth=-1,
                **mock_functions
            )
        
        # Depth too large
        with pytest.raises(ValueError, match="depth must be between 0 and 5"):
            await traverse_knowledge_graph_paginated(
                mock_graphiti,
                start_node_uuid="N1",
                depth=6,
                **mock_functions
            )
    
    @pytest.mark.asyncio
    async def test_first_page_returns_root_and_cursor(self, mock_graphiti, mock_functions):
        """Test first page returns root node and cursor info."""
        with patch("src.tools.traverse_wrapper.advance_bfs") as mock_advance:
            # Mock advance_bfs to return flat structure and indicate more pages
            mock_advance.return_value = (
                {
                    "start": "N1",
                    "nodes": {
                        "N1": {"uuid": "N1", "name": "Node N1"},
                        "N2": {"uuid": "N2", "name": "Node N2"}
                    },
                    "edges": [
                        {"id": "E:N1:N2:0", "source": "N1", "target": "N2", "depth": 1, "order": 0}
                    ]
                },
                True,  # has_more
                150,   # estimated_tokens
            )
            
            result = await traverse_knowledge_graph_paginated(
                mock_graphiti,
                start_node_uuid="N1",
                depth=2,
                **mock_functions
            )
            
            # Check flat structure is returned
            assert result["start"] == "N1"
            assert "N1" in result["nodes"]
            assert result["nodes"]["N1"]["uuid"] == "N1"
            
            # Check edges are included (flat)
            assert len(result["edges"]) == 1
            assert result["edges"][0]["source"] == "N1"
            assert result["edges"][0]["target"] == "N2"
            
            # Check cursor is present
            assert "cursor" in result
            assert result["cursor"]["has_more"] is True
            assert "token" in result["cursor"]
            assert "expires_at" in result["cursor"]
            
            # Check usage stats
            assert result["usage"]["estimated_tokens"] == 150
    
    @pytest.mark.asyncio
    async def test_complete_traversal_no_cursor(self, mock_graphiti, mock_functions):
        """Test complete traversal returns no cursor."""
        with patch("src.tools.traverse_wrapper.advance_bfs") as mock_advance:
            # Mock advance_bfs with flat structure indicating traversal complete
            mock_advance.return_value = (
                {
                    "start": "N1",
                    "nodes": {
                        "N1": {"uuid": "N1", "name": "Node N1"}
                    },
                    "edges": []
                },
                False,  # has_more = False
                50,
            )
            
            result = await traverse_knowledge_graph_paginated(
                mock_graphiti,
                start_node_uuid="N1",
                depth=1,
                **mock_functions
            )
            
            # Check cursor indicates completion
            assert result["cursor"]["has_more"] is False
            assert "token" not in result["cursor"]
    
    @pytest.mark.asyncio
    async def test_resume_with_valid_cursor(self, mock_graphiti, mock_functions):
        """Test resuming traversal with valid cursor."""
        with patch("src.tools.traverse_wrapper._session_store") as mock_store:
            with patch("src.tools.traverse_wrapper.advance_bfs") as mock_advance:
                # Setup mock session
                mock_session = MagicMock()
                mock_session.query_hash = "N1:2"
                mock_session.root_uuid = "N1"
                
                # Mock token verification and session loading
                mock_store.verify_token = AsyncMock(return_value={
                    "sid": "session-123",
                    "qh": "N1:2",
                })
                mock_store.load_session = AsyncMock(return_value=mock_session)
                mock_store.delete_session = AsyncMock()  # Add this mock
                
                # Mock advance_bfs for continued traversal with flat structure
                mock_advance.return_value = (
                    {
                        "start": "N1",  # Start UUID is always present
                        "nodes": {
                            "N3": {"uuid": "N3", "name": "Node N3"}
                        },
                        "edges": [
                            {"id": "E:N1:N3:1", "source": "N1", "target": "N3", "depth": 2, "order": 1}
                        ]
                    },
                    False,  # Last page
                    75,
                )
                
                result = await traverse_knowledge_graph_paginated(
                    mock_graphiti,
                    cursor_token="valid-token",
                    **mock_functions
                )
                
                # Should still have start UUID
                assert result["start"] == "N1"
                
                # Should have new nodes from continuation
                assert "N3" in result["nodes"]
                
                # Should have edges from continuation (flat)
                assert len(result["edges"]) == 1
                assert result["edges"][0]["target"] == "N3"
                
                # Should indicate completion
                assert result["cursor"]["has_more"] is False
    
    @pytest.mark.asyncio
    async def test_invalid_cursor_raises_error(self, mock_graphiti, mock_functions):
        """Test invalid cursor raises InvalidCursor error."""
        with patch("src.tools.traverse_wrapper._session_store") as mock_store:
            # Mock token verification to raise error
            mock_store.verify_token = AsyncMock(side_effect=Exception("Bad token"))
            
            with pytest.raises(InvalidCursor):
                await traverse_knowledge_graph_paginated(
                    mock_graphiti,
                    cursor_token="invalid-token",
                    **mock_functions
                )
    
    @pytest.mark.asyncio
    async def test_expired_cursor_raises_error(self, mock_graphiti, mock_functions):
        """Test expired cursor raises CursorExpired error."""
        with patch("src.tools.traverse_wrapper._session_store") as mock_store:
            # Mock token verification to raise CursorExpired
            mock_store.verify_token = AsyncMock(side_effect=CursorExpired("Token expired"))
            
            with pytest.raises(CursorExpired):
                await traverse_knowledge_graph_paginated(
                    mock_graphiti,
                    cursor_token="expired-token",
                    **mock_functions
                )
    
    @pytest.mark.asyncio
    async def test_session_not_found_raises_error(self, mock_graphiti, mock_functions):
        """Test missing session raises SessionNotFound error."""
        with patch("src.tools.traverse_wrapper._session_store") as mock_store:
            # Mock successful token verification but no session
            mock_store.verify_token = AsyncMock(return_value={
                "sid": "missing-session",
                "qh": "N1:2",
            })
            mock_store.load_session = AsyncMock(return_value=None)
            
            with pytest.raises(SessionNotFound):
                await traverse_knowledge_graph_paginated(
                    mock_graphiti,
                    cursor_token="token-for-missing-session",
                    **mock_functions
                )
    
    @pytest.mark.asyncio
    async def test_query_mismatch_raises_error(self, mock_graphiti, mock_functions):
        """Test query hash mismatch raises QueryMismatch error."""
        with patch("src.tools.traverse_wrapper._session_store") as mock_store:
            # Setup mock session with different query hash
            mock_session = MagicMock()
            mock_session.query_hash = "N1:3"  # Different from token
            
            mock_store.verify_token = AsyncMock(return_value={
                "sid": "session-123",
                "qh": "N1:2",  # Different from session
            })
            mock_store.load_session = AsyncMock(return_value=mock_session)
            
            with pytest.raises(QueryMismatch):
                await traverse_knowledge_graph_paginated(
                    mock_graphiti,
                    cursor_token="mismatched-token",
                    **mock_functions
                )
    
    # Removed test_extended_function_* tests as traverse_knowledge_graph_extended is deprecated
    # The functionality is now directly handled by traverse_knowledge_graph_impl calling traverse_knowledge_graph_paginated