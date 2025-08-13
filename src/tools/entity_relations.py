"""Entity relations functionality for Graphiti MCP server."""

from typing import Any, cast, TypedDict
import logging
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge


class ErrorResponse(TypedDict):
    error: str

logger = logging.getLogger(__name__)


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an EntityEdge as a fact result.
    
    Args:
        edge: The EntityEdge to format
        
    Returns:
        A dictionary containing the formatted fact
    """
    return {
        'uuid': edge.uuid,
        'name': edge.name,
        'fact': edge.fact,
        'created_at': edge.created_at.isoformat() if edge.created_at else None,
        'valid_at': edge.valid_at.isoformat() if edge.valid_at else None,
        'invalid_at': edge.invalid_at.isoformat() if edge.invalid_at else None,
        'confidence': getattr(edge, 'confidence', None),
        'source_uuid': edge.source_node_uuid,
        'target_uuid': edge.target_node_uuid,
        'episodes': edge.episodes if hasattr(edge, 'episodes') else [],
    }


async def get_entity_relations(
    graphiti_client: Graphiti | None,
    entity_uuid: str,
) -> list[dict[str, Any]] | ErrorResponse:
    """Get all relationships (edges) connected to a specific entity.

    Args:
        graphiti_client: The Graphiti client instance
        entity_uuid: UUID of the entity node to get relationships for
        
    Returns:
        A list of formatted edges or an ErrorResponse
    """
    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get all edges connected to this entity using the EntityEdge class method
        entity_edges = await EntityEdge.get_by_node_uuid(client.driver, entity_uuid)

        if not entity_edges:
            return []

        # Format all edges for response
        formatted_edges = [format_fact_result(edge) for edge in entity_edges]
        
        return formatted_edges
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity relations: {error_msg}')
        return ErrorResponse(error=f'Error getting entity relations: {error_msg}')