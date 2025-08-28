"""Flat response format for traverse_knowledge_graph."""

from typing import Any, Dict
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge


def format_node_flat(node: EntityNode) -> Dict[str, Any]:
    """Format an EntityNode for flat structure.
    
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


def format_edge_flat(
    edge: EntityEdge,
    depth: int = 0,
    order: int = 0,
    edge_id: str | None = None,
) -> Dict[str, Any]:
    """Format an EntityEdge for flat structure.
    
    Args:
        edge: The EntityEdge to format
        depth: Distance from start node
        order: Order index for stable sorting
        edge_id: Optional edge ID, will be generated if not provided
        
    Returns:
        A dictionary containing the formatted edge
    """
    # Generate edge ID if not provided
    if edge_id is None:
        # Use a composite of source and target UUIDs as ID
        edge_id = f"E:{edge.source_node_uuid}:{edge.target_node_uuid}"
    
    return {
        'id': edge_id,
        'type': edge.name,
        'fact': edge.fact,
        'source': edge.source_node_uuid,
        'target': edge.target_node_uuid,
        'episodes': edge.episodes if hasattr(edge, 'episodes') else [],
        'created_at': edge.created_at.isoformat() if edge.created_at else None,
        'valid_at': edge.valid_at.isoformat() if edge.valid_at else None,
        'invalid_at': edge.invalid_at.isoformat() if edge.invalid_at else None,
        'depth': depth,
        'order': order,
    }