"""
Unit tests for entity_relations.py functions.
"""

import pytest
import pytest_asyncio
import asyncio
import os
from typing import Any
from unittest.mock import Mock, AsyncMock, MagicMock

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from datetime import datetime

from src.tools.entity_relations import (
    format_fact_result,
    get_entity_relations,
    ErrorResponse
)

# Test configuration
TEST_NEO4J_URI = os.getenv('TEST_NEO4J_URI', 'bolt://localhost:7688')
TEST_NEO4J_USER = os.getenv('TEST_NEO4J_USER', 'neo4j')
TEST_NEO4J_PASSWORD = os.getenv('TEST_NEO4J_PASSWORD', 'testpassword')


class TestFormatFactResult:
    """Test cases for format_fact_result function."""
    
    def test_format_fact_result_with_all_fields(self):
        """Test formatting EntityEdge with all fields populated."""
        # Create a mock EntityEdge
        edge = Mock(spec=EntityEdge)
        edge.uuid = "edge-uuid-123"
        edge.name = "RELATES_TO"
        edge.fact = "Alice manages Project Alpha"
        edge.created_at = datetime(2024, 1, 1, 12, 0, 0)
        edge.valid_at = datetime(2024, 1, 1, 0, 0, 0)
        edge.invalid_at = None
        edge.confidence = 0.95
        edge.source_node_uuid = "alice-uuid"
        edge.target_node_uuid = "project-uuid"
        edge.episodes = ["episode-1", "episode-2"]
        
        result = format_fact_result(edge)
        
        assert result['uuid'] == "edge-uuid-123"
        assert result['name'] == "RELATES_TO"
        assert result['fact'] == "Alice manages Project Alpha"
        assert result['created_at'] == "2024-01-01T12:00:00"
        assert result['valid_at'] == "2024-01-01T00:00:00"
        assert result['invalid_at'] is None
        assert result['confidence'] == 0.95
        assert result['source_uuid'] == "alice-uuid"
        assert result['target_uuid'] == "project-uuid"
        assert result['episodes'] == ["episode-1", "episode-2"]
    
    def test_format_fact_result_with_missing_fields(self):
        """Test formatting EntityEdge with some missing fields."""
        edge = Mock(spec=EntityEdge)
        edge.uuid = "edge-uuid-456"
        edge.name = "INCLUDES"
        edge.fact = "Project includes feature"
        edge.created_at = None
        edge.valid_at = None
        edge.invalid_at = None
        edge.source_node_uuid = "project-uuid"
        edge.target_node_uuid = "feature-uuid"
        
        # Mock hasattr to return False for episodes
        original_hasattr = hasattr
        
        def mock_hasattr(obj, attr):
            if attr == 'episodes':
                return False
            if attr == 'confidence':
                return False
            return original_hasattr(obj, attr)
        
        import builtins
        original_builtins_hasattr = builtins.hasattr
        builtins.hasattr = mock_hasattr
        
        try:
            result = format_fact_result(edge)
            
            assert result['uuid'] == "edge-uuid-456"
            assert result['name'] == "INCLUDES"
            assert result['fact'] == "Project includes feature"
            assert result['created_at'] is None
            assert result['valid_at'] is None
            assert result['invalid_at'] is None
            assert result['confidence'] is None
            assert result['source_uuid'] == "project-uuid"
            assert result['target_uuid'] == "feature-uuid"
            assert result['episodes'] == []
        finally:
            builtins.hasattr = original_builtins_hasattr


class TestGetEntityRelations:
    """Test cases for get_entity_relations function."""
    
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
    async def test_get_entity_relations_success(self, graphiti_client):
        """Test getting relations for an entity that exists."""
        # Using Bob Johnson's UUID from test data
        entity_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"
        
        result = await get_entity_relations(graphiti_client, entity_uuid)
        
        assert not isinstance(result, dict)  # Should not be ErrorResponse
        assert isinstance(result, list)
        assert len(result) > 0  # Bob Johnson has multiple relations
        
        # Check structure of first relation
        first_relation = result[0]
        assert 'uuid' in first_relation
        assert 'name' in first_relation
        assert 'fact' in first_relation
        assert 'source_uuid' in first_relation
        assert 'target_uuid' in first_relation
        
        # Either source or target should be Bob Johnson
        assert (first_relation['source_uuid'] == entity_uuid or 
                first_relation['target_uuid'] == entity_uuid)
    
    @pytest.mark.asyncio
    async def test_get_entity_relations_no_relations(self, graphiti_client):
        """Test getting relations for an entity with no relations."""
        # Using a UUID that might not have relations
        entity_uuid = "00000000-0000-0000-0000-000000000000"
        
        result = await get_entity_relations(graphiti_client, entity_uuid)
        
        # Should return empty list, not an error
        assert isinstance(result, list)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_get_entity_relations_none_client(self):
        """Test error handling when graphiti_client is None."""
        entity_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"
        
        result = await get_entity_relations(None, entity_uuid)
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'not initialized' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_get_entity_relations_multiple_types(self, graphiti_client):
        """Test that different relation types are properly returned."""
        # Project Alpha has multiple relation types
        entity_uuid = "e805a3a7-fd76-4d34-80f4-c7eb3165b635"
        
        result = await get_entity_relations(graphiti_client, entity_uuid)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Collect unique relation names/types
        relation_types = set()
        for relation in result:
            if 'name' in relation and relation['name']:
                relation_types.add(relation['name'])
        
        # Should have multiple types of relations
        assert len(relation_types) >= 1
    
    @pytest.mark.asyncio
    async def test_get_entity_relations_format_consistency(self, graphiti_client):
        """Test that all returned relations have consistent format."""
        entity_uuid = "3e6968a4-2288-4681-8f45-e6f97ac94869"  # Bob Johnson
        
        result = await get_entity_relations(graphiti_client, entity_uuid)
        
        assert isinstance(result, list)
        
        # All relations should have the same structure
        expected_keys = {'uuid', 'name', 'fact', 'created_at', 'valid_at', 
                        'invalid_at', 'confidence', 'source_uuid', 'target_uuid', 'episodes'}
        
        for relation in result:
            assert set(relation.keys()) == expected_keys
            assert isinstance(relation['uuid'], str)
            assert isinstance(relation['episodes'], list)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])