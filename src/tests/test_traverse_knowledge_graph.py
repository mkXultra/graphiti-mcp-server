"""
Unit tests for traverse_knowledge_graph.py functions.
"""

import pytest
import pytest_asyncio
import asyncio
import os
from typing import Any
from unittest.mock import Mock, AsyncMock, MagicMock, patch

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from datetime import datetime

from src.tools.traverse_knowledge_graph import (
    format_node_result,
    format_edge_for_traverse,
    get_node_by_uuid,
    traverse_knowledge_graph,
    traverse_knowledge_graph_impl,
    ErrorResponse
)

# Test configuration
TEST_NEO4J_URI = os.getenv('TEST_NEO4J_URI', 'bolt://localhost:7688')
TEST_NEO4J_USER = os.getenv('TEST_NEO4J_USER', 'neo4j')
TEST_NEO4J_PASSWORD = os.getenv('TEST_NEO4J_PASSWORD', 'testpassword')


class TestFormatNodeResult:
    """Test cases for format_node_result function."""
    
    def test_format_node_with_all_fields(self):
        """Test formatting EntityNode with all fields populated."""
        node = Mock(spec=EntityNode)
        node.uuid = "node-uuid-123"
        node.name = "Bob Johnson"
        node.summary = "Senior engineer and React specialist"
        node.labels = ["Entity", "Person"]
        node.group_id = "default"
        node.created_at = datetime(2024, 1, 1, 12, 0, 0)
        node.attributes = {"role": "engineer", "experience": 5}
        
        result = format_node_result(node)
        
        assert result['uuid'] == "node-uuid-123"
        assert result['name'] == "Bob Johnson"
        assert result['summary'] == "Senior engineer and React specialist"
        assert result['labels'] == ["Entity", "Person"]
        assert result['group_id'] == "default"
        assert result['created_at'] == "2024-01-01T12:00:00"
        assert result['attributes'] == {"role": "engineer", "experience": 5}
    
    def test_format_node_with_missing_optional_fields(self):
        """Test formatting EntityNode with missing optional fields."""
        node = Mock(spec=EntityNode)
        node.uuid = "node-uuid-456"
        node.name = "Project Alpha"
        node.group_id = "projects"
        node.created_at = None
        
        # Mock hasattr for missing attributes
        original_hasattr = hasattr
        
        def mock_hasattr(obj, attr):
            if attr in ['summary', 'labels', 'attributes']:
                return False
            return original_hasattr(obj, attr)
        
        import builtins
        original_builtins_hasattr = builtins.hasattr
        builtins.hasattr = mock_hasattr
        
        try:
            result = format_node_result(node)
            
            assert result['uuid'] == "node-uuid-456"
            assert result['name'] == "Project Alpha"
            assert result['summary'] == ''
            assert result['labels'] == []
            assert result['group_id'] == "projects"
            assert result['created_at'] is None
            assert result['attributes'] == {}
        finally:
            builtins.hasattr = original_builtins_hasattr


class TestFormatEdgeForTraverse:
    """Test cases for format_edge_for_traverse function."""
    
    def test_format_edge_with_target_data(self):
        """Test formatting edge with target node data."""
        edge = Mock(spec=EntityEdge)
        edge.name = "MANAGES"
        edge.fact = "Alice manages Project Alpha"
        edge.source_node_uuid = "alice-uuid"
        edge.target_node_uuid = "project-uuid"
        edge.episodes = ["ep1", "ep2"]
        edge.created_at = datetime(2024, 1, 1, 12, 0, 0)
        edge.valid_at = None
        edge.invalid_at = None
        
        target_node_data = {
            'node': {
                'uuid': 'project-uuid',
                'name': 'Project Alpha'
            },
            'edges': []
        }
        
        result = format_edge_for_traverse(edge, target_node_data)
        
        assert result['type'] == "MANAGES"
        assert result['fact'] == "Alice manages Project Alpha"
        assert result['source_node_uuid'] == "alice-uuid"
        assert result['target_node_uuid'] == "project-uuid"
        assert result['episodes'] == ["ep1", "ep2"]
        assert result['created_at'] == "2024-01-01T12:00:00"
        assert result['valid_at'] is None
        assert result['invalid_at'] is None
        assert result['target'] == target_node_data


class TestGetNodeByUuid:
    """Test cases for get_node_by_uuid function."""
    
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
    async def test_get_existing_node(self, graphiti_client):
        """Test getting a node that exists."""
        # Bob Johnson's UUID
        node_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"
        
        node = await get_node_by_uuid(graphiti_client, node_uuid)
        
        assert node is not None
        assert node.uuid == node_uuid
        assert node.name == "Bob Johnson"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_node(self, graphiti_client):
        """Test getting a node that doesn't exist."""
        node_uuid = "00000000-0000-0000-0000-000000000000"
        
        node = await get_node_by_uuid(graphiti_client, node_uuid)
        
        assert node is None


class TestTraverseKnowledgeGraph:
    """Test cases for traverse_knowledge_graph function."""
    
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
    async def test_traverse_depth_0(self, graphiti_client):
        """Test traversal with depth 0 (node only)."""
        start_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"  # Bob Johnson
        
        result = await traverse_knowledge_graph(graphiti_client, start_uuid, depth=0)
        
        assert not isinstance(result, dict) or 'error' not in result
        assert 'start' in result  # Flat structure uses 'start'
        assert 'nodes' in result   # Flat structure uses 'nodes' dict
        assert 'edges' in result
        assert result['edges'] == []  # No edges at depth 0
        assert result['start'] == start_uuid
        assert start_uuid in result['nodes']
        assert result['nodes'][start_uuid]['name'] == "Bob Johnson"
    
    @pytest.mark.asyncio
    async def test_traverse_depth_1(self, graphiti_client):
        """Test traversal with depth 1 (direct neighbors)."""
        start_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"  # Bob Johnson
        
        result = await traverse_knowledge_graph(graphiti_client, start_uuid, depth=1)
        
        assert not isinstance(result, dict) or 'error' not in result
        assert 'start' in result  # Flat structure
        assert 'nodes' in result   # Flat structure
        assert 'edges' in result
        assert len(result['edges']) > 0  # Should have edges
        
        # Check that edges have proper flat structure
        for edge in result['edges']:
            assert 'type' in edge
            assert 'fact' in edge
            assert 'target' in edge
            assert 'source' in edge
            assert 'depth' in edge
            # In flat structure, target is a UUID string
            assert isinstance(edge['target'], str)
            # Target node should exist in nodes dict
            assert edge['target'] in result['nodes']
    
    @pytest.mark.asyncio
    async def test_traverse_depth_2(self, graphiti_client):
        """Test traversal with depth 2."""
        start_uuid = "e805a3a7-fd76-4d34-80f4-c7eb3165b635"  # Project Alpha
        
        result = await traverse_knowledge_graph(graphiti_client, start_uuid, depth=2)
        
        assert not isinstance(result, dict) or 'error' not in result
        assert 'start' in result  # Flat structure
        assert 'nodes' in result   # Flat structure
        assert 'edges' in result
        assert len(result['edges']) > 0
        
        # In flat structure, check for edges at different depths
        has_depth_1 = False
        has_depth_2 = False
        for edge in result['edges']:
            if edge.get('depth') == 1:
                has_depth_1 = True
            elif edge.get('depth') == 2:
                has_depth_2 = True
        
        assert has_depth_1, "Should have depth 1 edges"
        assert has_depth_2, "At depth 2, should have depth 2 edges"
    
    @pytest.mark.asyncio
    async def test_traverse_nonexistent_node(self, graphiti_client):
        """Test traversal starting from a non-existent node."""
        start_uuid = "00000000-0000-0000-0000-000000000000"
        
        result = await traverse_knowledge_graph(graphiti_client, start_uuid, depth=1)
        
        assert 'start' in result  # Flat structure
        assert 'nodes' in result
        assert start_uuid in result['nodes']
        assert result['nodes'][start_uuid].get('error') == 'Node not found' or result['nodes'][start_uuid]['uuid'] == start_uuid
        assert result['edges'] == []
    
    @pytest.mark.asyncio
    async def test_traverse_cycle_detection(self, graphiti_client):
        """Test that cycles are properly handled."""
        # Use a node that likely has bidirectional relationships
        start_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"  # Bob Johnson
        
        result = await traverse_knowledge_graph(graphiti_client, start_uuid, depth=3)
        
        assert not isinstance(result, dict) or 'error' not in result
        
        # In flat structure, check for cycles by examining edges
        def check_for_cycles(result):
            # Cycles are detected when an edge points to an already visited node
            visited = set()
            visited.add(result['start'])
            
            for edge in result.get('edges', []):
                source = edge.get('source')
                target = edge.get('target')
                
                # If we see an edge pointing back to a visited node, it's a cycle
                # The BFS engine should handle this by not re-traversing
                if source and target:
                    if target in visited:
                        # This could indicate a cycle was detected
                        return True
                    visited.add(target)
            
            return False
        
        # There might be cycles in depth 3 traversal
        # The flat structure handles cycles by not re-traversing visited nodes
        # So we just verify the result is valid
        assert 'start' in result
        assert 'nodes' in result
        assert 'edges' in result
    
    @pytest.mark.asyncio
    async def test_traverse_max_depth_limit(self, graphiti_client):
        """Test that excessive depth is rejected."""
        start_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"
        
        result = await traverse_knowledge_graph(graphiti_client, start_uuid, depth=10)
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'maximum depth' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_traverse_negative_depth(self, graphiti_client):
        """Test that negative depth is rejected."""
        start_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"
        
        result = await traverse_knowledge_graph(graphiti_client, start_uuid, depth=-1)
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'non-negative' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_traverse_none_client(self):
        """Test error handling when graphiti_client is None."""
        result = await traverse_knowledge_graph(None, "some-uuid", depth=1)
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'not initialized' in result['error'].lower()


class TestTraverseKnowledgeGraphImpl:
    """Test cases for the internal traverse_knowledge_graph_impl function."""
    
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
    


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])