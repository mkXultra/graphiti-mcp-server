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
    start_node_uuid: str | None = None,
    depth: int = 1,
    cursor_token: str | None = None,
) -> dict[str, Any]:
    """Internal implementation of traverse_knowledge_graph with cursor-based pagination.
    
    This implementation now uses the paginated version for all traversals.
    
    Args:
        graphiti_client: The Graphiti client instance
        start_node_uuid: UUID of the node to start traversal from (required for initial call)
        depth: Depth of traversal (0 = node only, 1 = direct relations, etc.)
        cursor_token: Optional cursor token for resuming a paginated traversal
        
    Returns:
        A dictionary containing the node and its edges with nested structure.
        May include a 'cursor' field if pagination is needed.
    """
    from src.tools.traverse_wrapper import traverse_knowledge_graph_paginated
    from src.tools.traverse_wrapper import CursorExpired, InvalidCursor, SessionNotFound, QueryMismatch
    
    try:
        return await traverse_knowledge_graph_paginated(
            graphiti_client,
            start_node_uuid=start_node_uuid,
            depth=depth,
            cursor_token=cursor_token,
            format_node_result=format_node_result,
            format_edge_for_traverse=format_edge_for_traverse,
            get_node_by_uuid=get_node_by_uuid,
        )
    except CursorExpired as e:
        return {'error': f'CURSOR_EXPIRED: {str(e)}'}
    except InvalidCursor as e:
        return {'error': f'INVALID_CURSOR: {str(e)}'}
    except SessionNotFound as e:
        return {'error': f'SESSION_NOT_FOUND: {str(e)}'}
    except QueryMismatch as e:
        return {'error': f'QUERY_MISMATCH: {str(e)}'}
    except ValueError as e:
        return {'error': f'INVALID_ARGUMENT: {str(e)}'}
    except Exception as e:
        logger.error(f'Error traversing knowledge graph: {str(e)}')
        return {'error': f'Error traversing knowledge graph: {str(e)}'}


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