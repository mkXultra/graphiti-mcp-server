"""BFS engine for cursor-based graph traversal."""

import logging
from typing import Any, Dict, List, Tuple, Optional
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge

from .session_store import Frame, TraverseSession
from .token_budget import TokenBudget, estimate_tokens
from .format_flat import format_node_flat, format_edge_flat

logger = logging.getLogger(__name__)

# Edge ordering functions for stable sorting
EDGE_ORDER = {
    "uuid": lambda e: (
        getattr(e, "name", ""),
        getattr(e, "target_node_uuid", None) or getattr(e, "source_node_uuid", None)
    ),
    "type_then_uuid": lambda e: (
        getattr(e, "name", ""),
        getattr(e, "created_at", None),
        getattr(e, "target_node_uuid", None) or getattr(e, "source_node_uuid", None)
    ),
    "created_at_then_uuid": lambda e: (
        getattr(e, "created_at", None),
        getattr(e, "name", ""),
        getattr(e, "target_node_uuid", None) or getattr(e, "source_node_uuid", None)
    ),
}


async def advance_bfs(
    sess: TraverseSession,
    graphiti_client: Graphiti,
    *,
    format_node_result,
    format_edge_for_traverse,
    get_node_by_uuid,
    budget: Optional[TokenBudget] = None
) -> Tuple[Dict[str, Any], bool, int]:
    """Advance BFS traversal by one page.
    
    Args:
        sess: The traversal session with current state
        graphiti_client: The Graphiti client instance
        format_node_result: Function to format node results (unused, kept for compatibility)
        format_edge_for_traverse: Function to format edge results (unused, kept for compatibility)
        get_node_by_uuid: Function to get node by UUID
        budget: Optional token budget (defaults to new TokenBudget)
        
    Returns:
        Tuple of (partial_result, has_more, estimated_tokens)
            - partial_result: The graph data for this page
            - has_more: Whether there are more pages
            - estimated_tokens: Number of tokens used
    """
    if budget is None:
        budget = TokenBudget()
    
    # Initialize result with flat structure
    result: Dict[str, Any] = {
        "start": sess.root_uuid,
        "nodes": {},
        "edges": []
    }
    
    # First page: add root node to nodes dict
    if not sess.visited:
        sess.visited = [sess.root_uuid]
        root = await get_node_by_uuid(graphiti_client, sess.root_uuid)
        
        if root is None:
            result["nodes"][sess.root_uuid] = {"uuid": sess.root_uuid, "error": "Node not found"}
        else:
            result["nodes"][sess.root_uuid] = format_node_flat(root)
        
        # Initialize frontier with root if we need to traverse
        if sess.max_depth > 0:
            sess.frontier.append(Frame(sess.root_uuid, sess.max_depth, 0))
        
        # Early return if no traversal needed
        if sess.max_depth == 0:
            return result, False, estimate_tokens(result)
    
    # Process frontier queue
    while sess.frontier:
        frame = sess.frontier.pop(0)  # Dequeue from front
        
        # Get edges for current node
        try:
            edges = await EntityEdge.get_by_node_uuid(graphiti_client.driver, frame.node_uuid)
        except Exception as e:
            logger.error(f"Error getting edges for node {frame.node_uuid}: {str(e)}")
            edges = []
        
        if not edges:
            continue  # No edges, move to next frame
        
        # Sort edges for stable ordering
        key_fn = EDGE_ORDER.get(sess.edge_ordering, EDGE_ORDER["uuid"])
        edges_sorted = sorted(edges, key=key_fn)
        
        # Process edges starting from where we left off
        i = frame.next_edge_index
        while i < len(edges_sorted):
            edge = edges_sorted[i]
            
            # Determine target node
            if edge.source_node_uuid == frame.node_uuid:
                target_uuid = edge.target_node_uuid
            else:
                target_uuid = edge.source_node_uuid
            
            # Calculate depth from start (max_depth - depth_remaining + 1)
            current_depth = sess.max_depth - frame.depth_remaining + 1
            
            # Generate edge ID
            edge_id = f"E:{edge.source_node_uuid}:{edge.target_node_uuid}:{sess.yielded_edges}"
            
            # Format edge with metadata
            edge_obj = format_edge_flat(
                edge,
                depth=current_depth,
                order=sess.yielded_edges,
                edge_id=edge_id
            )
            
            # Add target node to nodes dict if not visited
            if target_uuid not in sess.visited:
                target_node = await get_node_by_uuid(graphiti_client, target_uuid)
                if target_node is None:
                    # Only add to nodes dict if this edge will be included
                    temp_nodes = {target_uuid: {"uuid": target_uuid, "error": "Node not found"}}
                else:
                    temp_nodes = {target_uuid: format_node_flat(target_node)}
                
                # Check if we can add this edge and node within budget
                temp_result = {
                    "nodes": {**result["nodes"], **temp_nodes},
                    "edges": result["edges"] + [edge_obj]
                }
                
                if budget.can_add_edge(result, edge_obj) and estimate_tokens(temp_result) <= budget.limit:
                    # Add the node and edge
                    result["nodes"].update(temp_nodes)
                    result["edges"].append(edge_obj)
                    sess.yielded_edges += 1
                    
                    # Mark as visited and add to frontier if needed
                    sess.visited.append(target_uuid)
                    if frame.depth_remaining > 1:
                        sess.frontier.append(
                            Frame(target_uuid, frame.depth_remaining - 1, 0)
                        )
                    i += 1
                else:
                    # Budget exceeded - save position and return
                    frame.next_edge_index = i
                    sess.frontier.insert(0, frame)  # Put frame back at front
                    est = estimate_tokens(result)
                    return result, True, est
            else:
                # Target already visited - just add the edge
                if budget.can_add_edge(result, edge_obj):
                    result["edges"].append(edge_obj)
                    sess.yielded_edges += 1
                    i += 1
                else:
                    # Budget exceeded - save position and return
                    frame.next_edge_index = i
                    sess.frontier.insert(0, frame)  # Put frame back at front
                    est = estimate_tokens(result)
                    return result, True, est
        
        # Finished processing all edges for this frame
        # Frame is discarded (not put back in frontier)
    
    # Frontier is empty - traversal complete
    est = estimate_tokens(result)
    return result, False, est