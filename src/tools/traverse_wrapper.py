"""Cursor-based pagination wrapper for traverse_knowledge_graph."""

import time
import uuid
import logging
from typing import Any, Dict, Optional, cast
from graphiti_core import Graphiti

from .session_store import (
    TraverseSession,
    Frame,
    SessionStore,
    CursorExpired,
    InvalidCursor,
    SessionNotFound,
    QueryMismatch,
)
from .engine_bfs import advance_bfs
from .token_budget import TokenBudget, estimate_tokens
from .traverse_knowledge_graph import (
    format_node_result,
    format_edge_for_traverse,
    get_node_by_uuid,
    traverse_knowledge_graph_impl,
)

logger = logging.getLogger(__name__)

# Global session store (in production, use Redis or similar)
_session_store = SessionStore()


async def traverse_knowledge_graph_paginated(
    graphiti_client: Graphiti,
    start_node_uuid: Optional[str] = None,
    depth: int = 1,
    cursor_token: Optional[str] = None,
    *,
    format_node_result,
    format_edge_for_traverse,
    get_node_by_uuid,
) -> Dict[str, Any]:
    """Traverse knowledge graph with cursor-based pagination.
    
    This wraps the BFS engine with session management and cursor tokens.
    
    Args:
        graphiti_client: The Graphiti client instance
        start_node_uuid: UUID of the starting node (required for initial call)
        depth: Depth of traversal (0-5)
        cursor_token: Token from previous page (for continuation)
        format_node_result: Function to format node results
        format_edge_for_traverse: Function to format edge results
        get_node_by_uuid: Function to get node by UUID
        
    Returns:
        Dictionary with:
            - node: Root node information (first page only)
            - edges: Edges processed in this page
            - cursor: Pagination info (token, has_more, expires_at)
            - usage: Token usage statistics
            
    Raises:
        CursorExpired: Token has expired
        InvalidCursor: Token is malformed or invalid
        SessionNotFound: Session does not exist
        QueryMismatch: Query parameters don't match original
        ValueError: Invalid arguments
    """
    
    # Handle cursor continuation
    if cursor_token:
        try:
            payload = await _session_store.verify_token(cursor_token)
        except CursorExpired:
            raise
        except Exception:
            raise InvalidCursor("Invalid or malformed cursor token")
        
        session_id = payload["sid"]
        sess = await _session_store.load_session(session_id)
        
        if not sess:
            raise SessionNotFound("Session not found or expired")
        
        # Verify query hash matches
        if payload.get("qh") != sess.query_hash:
            raise QueryMismatch("Query parameters don't match original request")
    
    # Handle new traversal
    else:
        if not start_node_uuid:
            raise ValueError("start_node_uuid is required for initial traversal")
        
        if depth < 0 or depth > 5:
            raise ValueError("depth must be between 0 and 5")
        
        # Create new session
        session_id = str(uuid.uuid4())
        sess = TraverseSession(
            root_uuid=start_node_uuid,
            max_depth=depth,
            strategy="bfs",
            edge_ordering="uuid",  # Could be configurable
            query_hash=f"{start_node_uuid}:{depth}",
            frontier=[],
            visited=[],
            yielded_edges=0,
            started_at=time.time(),
            expires_at=time.time() + 3600,  # 1 hour TTL
        )
    
    # Advance the traversal by one page
    partial_result, has_more, est_tokens = await advance_bfs(
        sess,
        graphiti_client,
        format_node_result=format_node_result,
        format_edge_for_traverse=format_edge_for_traverse,
        get_node_by_uuid=get_node_by_uuid,
        budget=TokenBudget(),  # Uses default 20,000 token limit
    )
    
    # Build response with flat structure
    response: Dict[str, Any] = {
        "start": partial_result.get("start", sess.root_uuid),
        "nodes": partial_result.get("nodes", {}),
        "edges": partial_result.get("edges", []),
        "usage": {"estimated_tokens": est_tokens},
    }
    
    # Handle pagination
    if has_more:
        # Save session for next page
        await _session_store.save_session(session_id, sess)
        
        # Issue new token
        token_info = await _session_store.issue_token(
            session_id,
            sess.query_hash,
            ttl_seconds=600,  # 10 minute token TTL
        )
        
        response["cursor"] = {
            "token": token_info["token"],
            "has_more": True,
            "expires_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(token_info["expires_at"])
            ),
        }
    else:
        # Traversal complete - clean up session
        if cursor_token:  # Only delete if we were using a session
            await _session_store.delete_session(session_id)
        
        response["cursor"] = {"has_more": False}
    
    return response


# Commented out as traverse_knowledge_graph_impl now directly calls traverse_knowledge_graph_paginated
# async def traverse_knowledge_graph_extended(
#     graphiti_client: Optional[Graphiti],
#     start_node_uuid: Optional[str] = None,
#     depth: int = 1,
#     cursor_token: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """Extended traverse_knowledge_graph with backward compatibility.
#     
#     This function maintains the original API while adding cursor support.
#     When cursor_token is None, it performs the full traversal (legacy mode).
#     When cursor_token is provided, it uses pagination.
#     
#     Args:
#         graphiti_client: The Graphiti client instance
#         start_node_uuid: UUID of the starting node
#         depth: Depth of traversal (0-5)
#         cursor_token: Optional cursor for pagination
#         
#     Returns:
#         Dictionary with traversal results
#         If cursor_token is used, includes cursor field
#         Otherwise, returns legacy format
#     """
#     if graphiti_client is None:
#         return {"error": "Graphiti client not initialized"}
#     
#     client = cast(Graphiti, graphiti_client)
#     
#     # If no cursor, check if we should use legacy mode
#     if cursor_token is None and start_node_uuid:
#         # Check if the graph is small enough for single page
#         # This is a heuristic - could be made configurable
#         if depth <= 1:
#             # Use legacy implementation for shallow traversals
#             try:
#                 result = await traverse_knowledge_graph_impl(
#                     client,
#                     start_node_uuid,
#                     depth,
#                 )
#                 return result
#             except Exception as e:
#                 logger.error(f"Error in legacy traversal: {str(e)}")
#                 return {"error": f"Error traversing knowledge graph: {str(e)}"}
#     
#     # Use paginated implementation
#     try:
#         return await traverse_knowledge_graph_paginated(
#             client,
#             start_node_uuid=start_node_uuid,
#             depth=depth,
#             cursor_token=cursor_token,
#             format_node_result=format_node_result,
#             format_edge_for_traverse=format_edge_for_traverse,
#             get_node_by_uuid=get_node_by_uuid,
#         )
#     except (CursorExpired, InvalidCursor, SessionNotFound, QueryMismatch) as e:
#         # Return MCP-compatible error response
#         error_map = {
#             CursorExpired: ("CURSOR_EXPIRED", 410),
#             InvalidCursor: ("INVALID_CURSOR", 400),
#             SessionNotFound: ("SESSION_NOT_FOUND", 404),
#             QueryMismatch: ("QUERY_MISMATCH", 409),
#         }
#         error_name, _ = error_map.get(type(e), ("UNKNOWN_ERROR", 500))
#         return {"error": f"{error_name}: {str(e)}"}
#     except ValueError as e:
#         return {"error": f"INVALID_ARGUMENT: {str(e)}"}
#     except Exception as e:
#         logger.error(f"Error in paginated traversal: {str(e)}")
#         return {"error": f"Error traversing knowledge graph: {str(e)}"}