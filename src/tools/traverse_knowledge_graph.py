"""Traverse knowledge graph functionality for Graphiti MCP server."""

from typing import Any, cast, TypedDict
import logging
from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters


class ErrorResponse(TypedDict):
    error: str

logger = logging.getLogger(__name__)


def format_node_result(node: EntityNode) -> dict[str, Any]:
    """Format an EntityNode as a result dictionary.
    
    Args:
        node: The EntityNode to format
        
    Returns:
        A dictionary containing the formatted node
    """
    return {
        'uuid': node.uuid,
        'name': node.name,
        'summary': node.summary if hasattr(node, 'summary') else '',
        'labels': node.labels if hasattr(node, 'labels') else [],
        'group_id': node.group_id,
        'created_at': node.created_at.isoformat() if node.created_at else None,
        'attributes': node.attributes if hasattr(node, 'attributes') else {},
    }


def format_edge_for_traverse(edge: EntityEdge, target_node_data: dict[str, Any]) -> dict[str, Any]:
    """Format an EntityEdge for traverse result.
    
    Args:
        edge: The EntityEdge to format
        target_node_data: The traversed data of the target node
        
    Returns:
        A dictionary containing the formatted edge with target node data
    """
    return {
        'type': edge.name,
        'fact': edge.fact,
        'source_node_uuid': edge.source_node_uuid,
        'target_node_uuid': edge.target_node_uuid,
        'episodes': edge.episodes if hasattr(edge, 'episodes') else [],
        'created_at': edge.created_at.isoformat() if edge.created_at else None,
        'valid_at': edge.valid_at.isoformat() if edge.valid_at else None,
        'invalid_at': edge.invalid_at.isoformat() if edge.invalid_at else None,
        'target': target_node_data,
    }


async def get_node_by_uuid(
    graphiti_client: Graphiti,
    node_uuid: str,
) -> EntityNode | None:
    """Get a node by its UUID.
    
    Args:
        graphiti_client: The Graphiti client instance
        node_uuid: UUID of the node to retrieve
        
    Returns:
        The EntityNode or None if not found
    """
    try:
        # Get the node directly using the EntityNode class method
        node = await EntityNode.get_by_uuid(graphiti_client.driver, node_uuid)
        return node
    except Exception as e:
        logger.error(f'Error getting node by UUID {node_uuid}: {str(e)}')
        return None


async def traverse_knowledge_graph_impl(
    graphiti_client: Graphiti,
    start_node_uuid: str,
    depth: int = 1,
    visited_nodes: set[str] | None = None,
) -> dict[str, Any]:
    """Internal implementation of traverse_knowledge_graph with cycle detection.
    
    Args:
        graphiti_client: The Graphiti client instance
        start_node_uuid: UUID of the node to start traversal from
        depth: Depth of traversal (0 = node only, 1 = direct relations, etc.)
        visited_nodes: Set of already visited node UUIDs to prevent cycles
        
    Returns:
        A dictionary containing the node and its edges with nested structure
    """
    # Initialize visited_nodes if this is the top-level call
    if visited_nodes is None:
        visited_nodes = set()
    
    # Check for cycles
    if start_node_uuid in visited_nodes:
        # Return minimal info for already visited nodes to prevent infinite recursion
        node = await get_node_by_uuid(graphiti_client, start_node_uuid)
        if node is None:
            return {
                'node': {'uuid': start_node_uuid, 'error': 'Node not found'},
                'edges': [],
                'cyclic_reference': True,
            }
        return {
            'node': format_node_result(node),
            'edges': [],
            'cyclic_reference': True,
        }
    
    # Mark this node as visited
    visited_nodes.add(start_node_uuid)
    
    # Get the node information
    node = await get_node_by_uuid(graphiti_client, start_node_uuid)
    if node is None:
        return {
            'node': {'uuid': start_node_uuid, 'error': 'Node not found'},
            'edges': [],
        }
    
    result = {
        'node': format_node_result(node),
        'edges': [],
    }
    
    # Base case: if depth is 0, return only the node
    if depth == 0:
        return result
    
    # Get all edges connected to this node
    try:
        # Get edges where this node is the source
        entity_edges = await EntityEdge.get_by_node_uuid(graphiti_client.driver, start_node_uuid)
        
        if not entity_edges:
            return result
        
        # Process each edge
        for edge in entity_edges:
            # Determine the target node UUID
            # If this node is the source, target is the target_node_uuid
            # If this node is the target, target is the source_node_uuid
            if edge.source_node_uuid == start_node_uuid:
                target_uuid = edge.target_node_uuid
            else:
                target_uuid = edge.source_node_uuid
            
            # Recursively traverse the target node with reduced depth
            target_data = await traverse_knowledge_graph_impl(
                graphiti_client,
                target_uuid,
                depth - 1,
                visited_nodes.copy(),  # Pass a copy to allow revisiting in different paths
            )
            
            # Format the edge with target data
            edge_info = format_edge_for_traverse(edge, target_data)
            result['edges'].append(edge_info)
        
    except Exception as e:
        logger.error(f'Error getting edges for node {start_node_uuid}: {str(e)}')
    
    return result


async def traverse_knowledge_graph(
    graphiti_client: Graphiti | None,
    start_node_uuid: str,
    depth: int = 1,
) -> dict[str, Any] | ErrorResponse:
    """Traverse the knowledge graph starting from a specific node.
    
    Args:
        graphiti_client: The Graphiti client instance
        start_node_uuid: UUID of the node to start traversal from
        depth: Depth of traversal (0 = node only, 1 = direct relations, etc.)
        
    Returns:
        A dictionary containing the traversed graph structure or an ErrorResponse
    """
    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')
    
    if depth < 0:
        return ErrorResponse(error='Depth must be a non-negative integer')
    
    if depth > 5:
        return ErrorResponse(error='Maximum depth is 5 to prevent excessive data retrieval')
    
    try:
        # Use cast to help the type checker
        client = cast(Graphiti, graphiti_client)
        
        # Perform the traversal
        result = await traverse_knowledge_graph_impl(client, start_node_uuid, depth)
        
        return result
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error traversing knowledge graph: {error_msg}')
        return ErrorResponse(error=f'Error traversing knowledge graph: {error_msg}')