"""
Unit tests for graph functions (find_paths_between_entities and build_subgraph).
TDD approach - write tests first, then implement.
"""

import pytest
import pytest_asyncio
import asyncio
import os
from typing import Any, TypedDict
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver

# Import the functions we're going to test (they don't exist yet)
from src.tools.graph_functions import (
    find_paths_between_entities,
    build_subgraph,
    PathSearchResponse,
    SubgraphResponse
)

# Test configuration
TEST_NEO4J_URI = os.getenv('TEST_NEO4J_URI', 'bolt://localhost:7688')
TEST_NEO4J_USER = os.getenv('TEST_NEO4J_USER', 'neo4j')
TEST_NEO4J_PASSWORD = os.getenv('TEST_NEO4J_PASSWORD', 'testpassword')


class TestFindPathsBetweenEntities:
    """Test cases for find_paths_between_entities function."""
    
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
    async def test_find_direct_path(self, graphiti_client):
        """Test finding a direct path between two connected entities."""
        # Using real UUIDs from test data
        # Alice Smith -> Bob Johnson (they have direct relationship via COLLABORATES_WITH)
        from_uuid = "24678c17-db39-46fd-98b4-7febd3dee444"  # Alice Smith
        to_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"    # Bob Johnson
        
        result = await find_paths_between_entities(
            graphiti_client=graphiti_client,
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            max_depth=1,
            max_paths=10
        )
        
        # Assertions
        assert isinstance(result, dict)
        assert 'error' not in result
        assert 'message' in result
        assert 'paths' in result
        assert len(result['paths']) > 0
        
        # Check first path
        first_path = result['paths'][0]
        assert 'path_id' in first_path
        assert 'length' in first_path
        assert 'node_sequence' in first_path
        assert 'edge_sequence' in first_path
        assert first_path['length'] == 1  # Direct connection
        assert from_uuid in first_path['node_sequence']
        assert to_uuid in first_path['node_sequence']
        
        # Check node details
        assert 'node_details' in result
        assert from_uuid in result['node_details']
        assert to_uuid in result['node_details']
        assert result['node_details'][from_uuid]['name'] == "Alice Smith"
        assert result['node_details'][to_uuid]['name'] == "Bob Johnson"
        
        # Check edge details
        assert 'edge_details' in result
        assert len(result['edge_details']) > 0
    
    @pytest.mark.asyncio
    async def test_find_multi_hop_paths(self, graphiti_client):
        """Test finding paths with multiple hops."""
        # TypeScript -> Bob Johnson (connected through Project Alpha)
        from_uuid = "205a9630-2ef7-432d-988c-141ee86af5b8"  # TypeScript
        to_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"    # Bob Johnson
        
        result = await find_paths_between_entities(
            graphiti_client=graphiti_client,
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            max_depth=3,
            max_paths=20
        )
        
        assert isinstance(result, dict)
        assert 'error' not in result
        assert len(result['paths']) > 0
        
        # Should find paths of different lengths
        path_lengths = [p['length'] for p in result['paths']]
        assert min(path_lengths) >= 1
        assert max(path_lengths) <= 3
        
        # Verify all paths start and end correctly
        for path in result['paths']:
            assert path['node_sequence'][0] == from_uuid
            assert path['node_sequence'][-1] == to_uuid
            assert len(path['edge_sequence']) == path['length']
    
    @pytest.mark.asyncio
    async def test_no_path_exists(self, graphiti_client):
        """Test when no path exists between entities."""
        # Create mock UUIDs that don't exist or aren't connected
        from_uuid = "00000000-0000-0000-0000-000000000001"
        to_uuid = "00000000-0000-0000-0000-000000000002"
        
        result = await find_paths_between_entities(
            graphiti_client=graphiti_client,
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            max_depth=5,
            max_paths=10
        )
        
        # Should return empty paths or error
        assert isinstance(result, dict)
        if 'error' not in result:
            assert result['paths'] == []
            assert 'message' in result
            assert 'no path' in result['message'].lower() or 'not found' in result['message'].lower()
    
    @pytest.mark.asyncio
    async def test_max_depth_limit(self, graphiti_client):
        """Test that max_depth parameter limits the search."""
        from_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"  # Bob Johnson
        to_uuid = "e805a3a7-fd76-4d34-80f4-c7eb3165b635"    # Project Alpha
        
        # Test with depth 1
        result_depth_1 = await find_paths_between_entities(
            graphiti_client=graphiti_client,
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            max_depth=1,
            max_paths=100
        )
        
        # Test with depth 3
        result_depth_3 = await find_paths_between_entities(
            graphiti_client=graphiti_client,
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            max_depth=3,
            max_paths=100
        )
        
        # Should find more or equal paths with higher depth
        assert len(result_depth_3['paths']) >= len(result_depth_1['paths'])
        
        # All paths in depth_1 should be length 1
        for path in result_depth_1['paths']:
            assert path['length'] <= 1
        
        # Paths in depth_3 can be up to length 3
        for path in result_depth_3['paths']:
            assert path['length'] <= 3
    
    @pytest.mark.asyncio
    async def test_max_paths_limit(self, graphiti_client):
        """Test that max_paths parameter limits the number of results."""
        from_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"  # Bob Johnson
        to_uuid = "e805a3a7-fd76-4d34-80f4-c7eb3165b635"    # Project Alpha
        
        result = await find_paths_between_entities(
            graphiti_client=graphiti_client,
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            max_depth=3,
            max_paths=5
        )
        
        assert isinstance(result, dict)
        assert 'error' not in result
        assert len(result['paths']) <= 5
    
    @pytest.mark.asyncio
    async def test_error_handling_none_client(self):
        """Test error handling when graphiti_client is None."""
        result = await find_paths_between_entities(
            graphiti_client=None,
            from_uuid="uuid1",
            to_uuid="uuid2",
            max_depth=3,
            max_paths=10
        )
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'not initialized' in result['error'].lower()


class TestBuildSubgraph:
    """Test cases for build_subgraph function."""
    
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
    async def test_build_subgraph_single_entity(self, graphiti_client):
        """Test building subgraph with a single entity."""
        entity_uuids = ["3e6968a4-2288-4681-8f45-e6f97ac94869"]  # Bob Johnson
        
        result = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=entity_uuids,
            include_paths=False,
            max_hop=0
        )
        
        assert isinstance(result, dict)
        assert 'error' not in result
        assert 'message' in result
        assert 'subgraph' in result
        
        subgraph = result['subgraph']
        assert 'nodes' in subgraph
        assert 'edges' in subgraph
        assert 'adjacency_list' in subgraph
        
        # With max_hop=0, should only have the single entity
        assert len(subgraph['nodes']) == 1
        assert entity_uuids[0] in subgraph['nodes']
        
        # Statistics
        assert 'statistics' in result
        assert result['statistics']['node_count'] == 1
        assert result['statistics']['edge_count'] == 0
    
    @pytest.mark.asyncio
    async def test_build_subgraph_with_neighbors(self, graphiti_client):
        """Test building subgraph with max_hop=1 to include neighbors."""
        entity_uuids = ["3e6968a4-2288-4681-8f45-e6f97ac94869"]  # Bob Johnson
        
        result = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=entity_uuids,
            include_paths=False,
            max_hop=1
        )
        
        assert isinstance(result, dict)
        assert 'error' not in result
        
        subgraph = result['subgraph']
        # Should have more nodes than just the starting entity
        assert len(subgraph['nodes']) > 1
        assert entity_uuids[0] in subgraph['nodes']
        
        # Should have edges
        assert len(subgraph['edges']) > 0
        
        # Check adjacency list
        assert entity_uuids[0] in subgraph['adjacency_list']
        assert len(subgraph['adjacency_list'][entity_uuids[0]]) > 0
        
        # Statistics
        assert result['statistics']['node_count'] > 1
        assert result['statistics']['edge_count'] > 0
    
    @pytest.mark.asyncio
    async def test_build_subgraph_multiple_entities(self, graphiti_client):
        """Test building subgraph with multiple starting entities."""
        entity_uuids = [
            "3e6968a4-2288-4681-8f45-e6f97ac94869",  # Bob Johnson
            "e805a3a7-fd76-4d34-80f4-c7eb3165b635",  # Project Alpha
            "205a9630-2ef7-432d-988c-141ee86af5b8"   # TypeScript
        ]
        
        result = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=entity_uuids,
            include_paths=False,
            max_hop=1
        )
        
        assert isinstance(result, dict)
        assert 'error' not in result
        
        subgraph = result['subgraph']
        # All starting entities should be in the graph
        for uuid in entity_uuids:
            assert uuid in subgraph['nodes']
        
        # Should have a connected subgraph
        assert len(subgraph['edges']) > 0
        
        # Check node details have required fields
        for uuid, node in subgraph['nodes'].items():
            assert 'uuid' in node
            assert 'name' in node
            assert 'labels' in node
    
    @pytest.mark.asyncio
    async def test_build_subgraph_with_paths(self, graphiti_client):
        """Test building subgraph with paths_between_entities included."""
        entity_uuids = [
            "3e6968a4-2288-4681-8f45-e6f97ac94869",  # Bob Johnson
            "e805a3a7-fd76-4d34-80f4-c7eb3165b635"   # Project Alpha
        ]
        
        result = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=entity_uuids,
            include_paths=True,
            max_hop=1
        )
        
        assert isinstance(result, dict)
        assert 'error' not in result
        
        # Should have paths_between_entities when include_paths=True
        assert 'paths_between_entities' in result
        
        # Should have paths between the two entities
        paths = result['paths_between_entities']
        expected_key = f"{entity_uuids[0]}_to_{entity_uuids[1]}"
        assert expected_key in paths or f"{entity_uuids[1]}_to_{entity_uuids[0]}" in paths
        
        # Paths should have the expected structure
        if expected_key in paths:
            entity_paths = paths[expected_key]
            assert isinstance(entity_paths, list)
            if len(entity_paths) > 0:
                first_path = entity_paths[0]
                assert 'path_id' in first_path
                assert 'length' in first_path
                assert 'node_sequence' in first_path
                assert 'edge_sequence' in first_path
    
    @pytest.mark.asyncio
    async def test_build_subgraph_max_hop_expansion(self, graphiti_client):
        """Test that max_hop correctly expands the subgraph."""
        entity_uuids = ["3e6968a4-2288-4681-8f45-e6f97ac94869"]  # Bob Johnson
        
        # Test with different max_hop values
        result_hop_0 = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=entity_uuids,
            include_paths=False,
            max_hop=0
        )
        
        result_hop_1 = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=entity_uuids,
            include_paths=False,
            max_hop=1
        )
        
        result_hop_2 = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=entity_uuids,
            include_paths=False,
            max_hop=2
        )
        
        # Each increase in max_hop should include more or equal nodes
        nodes_0 = result_hop_0['statistics']['node_count']
        nodes_1 = result_hop_1['statistics']['node_count']
        nodes_2 = result_hop_2['statistics']['node_count']
        
        assert nodes_0 <= nodes_1
        assert nodes_1 <= nodes_2
        
        # hop_0 should have exactly 1 node (the starting entity)
        assert nodes_0 == 1
        
        # hop_1 and hop_2 should have more nodes
        assert nodes_1 > 1
        assert nodes_2 >= nodes_1
    
    @pytest.mark.asyncio
    async def test_build_subgraph_error_handling(self):
        """Test error handling for build_subgraph."""
        # Test with None client
        result = await build_subgraph(
            graphiti_client=None,
            entity_uuids=["uuid1"],
            include_paths=False,
            max_hop=1
        )
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'not initialized' in result['error'].lower()
        
    @pytest.mark.asyncio
    async def test_build_subgraph_empty_entity_list(self, graphiti_client):
        """Test building subgraph with empty entity list."""
        result = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=[],
            include_paths=False,
            max_hop=1
        )
        
        # Should handle empty list gracefully
        assert isinstance(result, dict)
        if 'error' not in result:
            assert result['statistics']['node_count'] == 0
            assert result['statistics']['edge_count'] == 0


# Integration test to verify both functions work together
class TestIntegration:
    """Integration tests for graph functions."""
    
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
    async def test_find_paths_then_build_subgraph(self, graphiti_client):
        """Test using find_paths results to build a subgraph."""
        # First, find paths
        from_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"  # Bob Johnson
        to_uuid = "e805a3a7-fd76-4d34-80f4-c7eb3165b635"    # Project Alpha
        
        path_result = await find_paths_between_entities(
            graphiti_client=graphiti_client,
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            max_depth=2,
            max_paths=5
        )
        
        assert 'error' not in path_result
        assert len(path_result['paths']) > 0
        
        # Collect all unique nodes from paths
        all_nodes = set()
        for path in path_result['paths']:
            all_nodes.update(path['node_sequence'])
        
        # Build subgraph with those nodes
        # Note: build_subgraph now only includes Entity nodes, not Episodic
        subgraph_result = await build_subgraph(
            graphiti_client=graphiti_client,
            entity_uuids=list(all_nodes),
            include_paths=True,
            max_hop=0  # Don't expand, just use the given nodes
        )
        
        assert 'error' not in subgraph_result
        
        # All Entity nodes from paths should be in subgraph (Episodic nodes are excluded)
        # Since find_paths_between_entities now only returns Entity nodes,
        # all nodes should be in the subgraph
        for node_uuid in all_nodes:
            assert node_uuid in subgraph_result['subgraph']['nodes']
        
        # Should have the relationships between them
        assert len(subgraph_result['subgraph']['edges']) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])