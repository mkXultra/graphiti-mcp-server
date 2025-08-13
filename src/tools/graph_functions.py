"""
Graph functions for finding paths between entities and building subgraphs.
TDD implementation - starting with stub functions.
"""

from typing import Any, TypedDict, cast
import logging
from datetime import datetime
from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode, get_entity_node_from_record
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.nodes.node_db_queries import ENTITY_NODE_RETURN
from graphiti_core.models.edges.edge_db_queries import ENTITY_EDGE_RETURN

logger = logging.getLogger(__name__)


# Type definitions for responses
class ErrorResponse(TypedDict):
    error: str


class PathResult(TypedDict):
    path_id: int
    length: int
    node_sequence: list[str]  # List of UUIDs
    edge_sequence: list[str]  # List of UUIDs


class PathSearchResponse(TypedDict):
    message: str
    paths: list[PathResult]
    node_details: dict[str, dict[str, Any]]  # UUID -> EntityNode attributes
    edge_details: dict[str, dict[str, Any]]  # UUID -> EntityEdge attributes
    metadata: dict[str, Any]


class SubgraphStatistics(TypedDict):
    node_count: int
    edge_count: int


class SubgraphData(TypedDict):
    nodes: dict[str, dict[str, Any]]  # UUID -> EntityNode attributes
    edges: list[dict[str, Any]]  # List of EntityEdge attributes
    adjacency_list: dict[str, list[str]]  # UUID -> list of adjacent node UUIDs


class SubgraphResponse(TypedDict):
    message: str
    subgraph: SubgraphData
    statistics: SubgraphStatistics
    paths_between_entities: dict[str, list[dict[str, Any]]]  # Optional
    metadata: dict[str, Any]


async def find_paths_between_entities(
    graphiti_client: Graphiti | None,
    from_uuid: str,
    to_uuid: str,
    max_depth: int = 5,
    max_paths: int = 10,
) -> PathSearchResponse | ErrorResponse:
    """
    Find paths between two entities in the knowledge graph.

    Args:
        graphiti_client: The Graphiti client instance
        from_uuid: UUID of the starting entity
        to_uuid: UUID of the target entity
        max_depth: Maximum path length to search (default: 5)
        max_paths: Maximum number of paths to return (default: 10)

    Returns:
        PathSearchResponse with found paths or ErrorResponse if error
    """
    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    try:
        # Execute Cypher query to find paths between Entity nodes
        # Note: max_depth must be part of the query string, not a parameter
        # Only Entity nodes with RELATES_TO edges (exclude Episodic)
        path_query = f"""
        MATCH p = (start:Entity {{uuid: $from_uuid}})-[:RELATES_TO*1..{max_depth}]-(end:Entity {{uuid: $to_uuid}})
        WITH p, length(p) as path_length
        ORDER BY path_length
        LIMIT $max_paths
        RETURN path_length,
               [n IN nodes(p) | n.uuid] as node_uuids,
               [r IN relationships(p) | r.uuid] as edge_uuids
        """

        path_result = await graphiti_client.driver.execute_query(
            path_query, from_uuid=from_uuid, to_uuid=to_uuid, max_paths=max_paths
        )

        # Parse path results
        paths = []
        all_node_uuids = set()
        all_edge_uuids = set()
        
        path_records = path_result.records if hasattr(path_result, "records") else path_result[0]
        
        for i, record in enumerate(path_records):
            node_uuids = record["node_uuids"]
            edge_uuids = record["edge_uuids"]
            path_length = record["path_length"]
            
            # Collect all unique nodes and edges
            all_node_uuids.update(node_uuids)
            all_edge_uuids.update(edge_uuids)
            
            # Create path result
            path_result = PathResult(
                path_id=i + 1,
                length=path_length,
                node_sequence=node_uuids,
                edge_sequence=edge_uuids,
            )
            paths.append(path_result)
        
        # Fetch node details using the same approach as build_subgraph
        node_details = {}
        if all_node_uuids:
            node_query = f"""
            MATCH (n:Entity)
            WHERE n.uuid IN $node_uuids
            RETURN {ENTITY_NODE_RETURN}
            """
            
            node_result = await graphiti_client.driver.execute_query(
                node_query, node_uuids=list(all_node_uuids)
            )
            node_records = node_result.records if hasattr(node_result, "records") else node_result[0]
            
            for record in node_records:
                try:
                    entity_node = get_entity_node_from_record(record)
                    
                    # Use model_dump with exclude to remove embeddings
                    exclude_dict = {
                        'name_embedding': True,
                        'summary_embedding': True,
                        'attributes': {'fact_embedding', 'name_embedding', 'summary_embedding'}
                    }
                    node_data = entity_node.model_dump(
                        mode='json',
                        exclude=exclude_dict
                    )
                    node_details[entity_node.uuid] = node_data
                except Exception as e:
                    logger.warning(f"Failed to process node: {e}")
        
        # Fetch edge details using the same approach as build_subgraph
        edge_details = {}
        if all_edge_uuids:
            edge_query = f"""
            MATCH (n)-[e:RELATES_TO]-(m)
            WHERE e.uuid IN $edge_uuids
            RETURN {ENTITY_EDGE_RETURN}
            """
            
            edge_result = await graphiti_client.driver.execute_query(
                edge_query, edge_uuids=list(all_edge_uuids)
            )
            edge_records = edge_result.records if hasattr(edge_result, "records") else edge_result[0]
            
            for record in edge_records:
                try:
                    entity_edge = get_entity_edge_from_record(record)
                    
                    # Use model_dump with exclude to remove embeddings
                    exclude_dict = {
                        'fact_embedding': True,
                        'attributes': {'fact_embedding', 'name_embedding', 'summary_embedding'}
                    }
                    edge_data = entity_edge.model_dump(
                        mode='json',
                        exclude=exclude_dict
                    )
                    edge_details[entity_edge.uuid] = edge_data
                except Exception as e:
                    logger.warning(f"Failed to process edge: {e}")

        # Prepare response
        if not paths:
            message = "No paths found between the specified entities"
        else:
            message = f"Found {len(paths)} path(s) between entities"

        return PathSearchResponse(
            message=message,
            paths=paths,
            node_details=node_details,
            edge_details=edge_details,
            metadata={
                "from_uuid": from_uuid,
                "to_uuid": to_uuid,
                "max_depth": max_depth,
                "max_paths": max_paths,
                "total_paths_found": len(paths),
            },
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error finding paths between entities: {error_msg}")
        return ErrorResponse(error=f"Error finding paths between entities: {error_msg}")


async def build_subgraph(
    graphiti_client: Graphiti | None,
    entity_uuids: list[str],
    include_paths: bool = True,
    max_hop: int = 1,
) -> SubgraphResponse | ErrorResponse:
    """
    Build a subgraph containing specified entities and their neighbors.

    Args:
        graphiti_client: The Graphiti client instance
        entity_uuids: List of entity UUIDs to include in the subgraph
        include_paths: Whether to include paths between entities (default: True)
        max_hop: Maximum distance from starting entities to include (default: 1)

    Returns:
        SubgraphResponse with the subgraph or ErrorResponse if error
    """
    if graphiti_client is None:
        return ErrorResponse(error="Graphiti client not initialized")

    if not entity_uuids:
        # Return empty subgraph for empty input
        return SubgraphResponse(
            message="Empty entity list provided",
            subgraph=SubgraphData(nodes={}, edges=[], adjacency_list={}),
            statistics=SubgraphStatistics(node_count=0, edge_count=0),
            paths_between_entities={},
            metadata={},
        )

    try:
        # Query to get subgraph with specified max_hop distance
        if max_hop == 0:
            # Get Entity nodes only (exclude Episodic nodes)
            entity_query = f"""
            MATCH (n:Entity)
            WHERE n.uuid IN $entity_uuids
            RETURN {ENTITY_NODE_RETURN}
            """
            
            # Get edges between Entity nodes only (RELATES_TO)
            edge_query = f"""
            MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
            WHERE n.uuid IN $entity_uuids AND m.uuid IN $entity_uuids AND n.uuid < m.uuid
            RETURN {ENTITY_EDGE_RETURN}
            """
        else:
            # Get Entity nodes and their neighbors within max_hop distance (exclude Episodic)
            entity_query = f"""
            UNWIND $entity_uuids AS start_uuid
            MATCH (start:Entity {{uuid: start_uuid}})
            OPTIONAL MATCH path = (start)-[*0..{max_hop}]-(n:Entity)
            WITH DISTINCT n
            WHERE n IS NOT NULL
            RETURN {ENTITY_NODE_RETURN}
            """
            
            # Get edges in the expanded neighborhood (Entity-Entity only)
            edge_query = f"""
            UNWIND $entity_uuids AS start_uuid
            MATCH (start:Entity {{uuid: start_uuid}})
            OPTIONAL MATCH path = (start)-[*0..{max_hop}]-(connected:Entity)
            WITH relationships(path) as rels
            UNWIND rels as e
            MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
            WITH DISTINCT e, n, m
            WHERE e IS NOT NULL
            RETURN {ENTITY_EDGE_RETURN}
            """

        # Execute queries for Entity nodes and edges only
        entity_result = await graphiti_client.driver.execute_query(
            entity_query, entity_uuids=entity_uuids
        )
        edge_result = await graphiti_client.driver.execute_query(
            edge_query, entity_uuids=entity_uuids
        )

        # Parse results
        entity_records = entity_result.records if hasattr(entity_result, "records") else entity_result[0]
        edge_records = edge_result.records if hasattr(edge_result, "records") else edge_result[0]

        nodes_dict = {}
        edges_list = []
        adjacency_list = {}
        edge_uuids_seen = set()

        # Process Entity nodes using Graphiti core functions
        if entity_records:
            for record in entity_records:
                try:
                    node = get_entity_node_from_record(record)
                    
                    # Convert to dict with JSON mode (auto-converts datetime to string) and exclude embeddings
                    # Exclude both top-level embeddings and embeddings within attributes
                    exclude_dict = {
                        'name_embedding': True,
                        'summary_embedding': True,
                        'attributes': {'fact_embedding', 'name_embedding', 'summary_embedding'}
                    }
                    node_data = node.model_dump(
                        mode='json',
                        exclude=exclude_dict
                    )
                    
                    nodes_dict[node.uuid] = node_data
                    adjacency_list[node.uuid] = []
                except Exception as e:
                    logger.warning(f"Failed to process entity node record: {e}")
        
        # Episodic nodes are intentionally excluded from the subgraph

        # Process edges using Graphiti core functions
        if edge_records:
            for record in edge_records:
                try:
                    edge = get_entity_edge_from_record(record)
                    
                    # Track unique edges
                    if edge.uuid not in edge_uuids_seen:
                        edge_uuids_seen.add(edge.uuid)
                        
                        # Convert to dict with JSON mode and exclude embeddings
                        # Exclude both top-level embeddings and embeddings within attributes
                        exclude_dict = {
                            'fact_embedding': True,
                            'attributes': {'fact_embedding', 'name_embedding', 'summary_embedding'}
                        }
                        edge_data = edge.model_dump(
                            mode='json',
                            exclude=exclude_dict
                        )
                        
                        edges_list.append(edge_data)
                        
                        # Update adjacency list
                        source_uuid = edge.source_node_uuid
                        target_uuid = edge.target_node_uuid
                        
                        if source_uuid in adjacency_list and target_uuid not in adjacency_list[source_uuid]:
                            adjacency_list[source_uuid].append(target_uuid)
                        if target_uuid in adjacency_list and source_uuid not in adjacency_list[target_uuid]:
                            adjacency_list[target_uuid].append(source_uuid)
                except Exception as e:
                    logger.warning(f"Failed to process edge record: {e}")

        # Build paths between entities if requested
        paths_between_entities = {}
        if include_paths and len(entity_uuids) > 1:
            # Calculate paths between each pair of specified entities
            for i, uuid1 in enumerate(entity_uuids):
                for uuid2 in entity_uuids[i + 1 :]:
                    if uuid1 in nodes_dict and uuid2 in nodes_dict:
                        # Use our find_paths function with limited depth
                        path_result = await find_paths_between_entities(
                            graphiti_client,
                            uuid1,
                            uuid2,
                            # Limit depth for performance
                            max_depth=min(3, max_hop * 2),
                            max_paths=5,
                        )
                        if (
                            not isinstance(path_result, dict)
                            or "error" not in path_result
                        ):
                            key = f"{uuid1}_to_{uuid2}"
                            paths_between_entities[key] = path_result.get("paths", [])

        # Prepare response
        return SubgraphResponse(
            message=f"Subgraph built with {len(nodes_dict)} nodes and {len(edges_list)} edges",
            subgraph=SubgraphData(
                nodes=nodes_dict, edges=edges_list, adjacency_list=adjacency_list
            ),
            statistics=SubgraphStatistics(
                node_count=len(nodes_dict), edge_count=len(edges_list)
            ),
            paths_between_entities=paths_between_entities,
            metadata={
                "entity_uuids": entity_uuids,
                "max_hop": max_hop,
                "include_paths": include_paths,
            },
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error building subgraph: {error_msg}")
        return ErrorResponse(error=f"Error building subgraph: {error_msg}")
