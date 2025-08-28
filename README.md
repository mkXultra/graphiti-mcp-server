# Graphiti MCP Server Extended

An extended version of the [Graphiti MCP Server](https://github.com/getzep/graphiti/tree/main/mcp_server) with additional graph manipulation tools for advanced knowledge graph operations.

This repository is forked from the original Graphiti MCP Server and adds custom tools for path finding and subgraph extraction, enabling more sophisticated graph analysis capabilities.

## 🆕 Additional Features

This extended version adds three powerful graph manipulation tools:

### 1. `find_paths_between_entities`
Find all paths between two entities in the knowledge graph.

**Features:**
- Discovers multiple paths between entities
- Configurable maximum path depth
- Returns detailed node and edge information
- Excludes Episodic nodes for cleaner knowledge paths
- Optimized for Entity-to-Entity relationships only

**Parameters:**
- `from_uuid`: Starting entity UUID
- `to_uuid`: Target entity UUID  
- `max_depth`: Maximum path length (default: 5)
- `max_paths`: Maximum number of paths to return (default: 10)

### 2. `build_subgraph`
Extract a subgraph centered around specified entities.

**Features:**
- Build subgraphs from multiple entity starting points
- Configurable expansion depth (max_hop)
- Optional path finding between entities in the subgraph
- Returns complete graph structure with adjacency lists
- Automatically excludes embedding fields for smaller response sizes

**Parameters:**
- `entity_uuids`: List of entity UUIDs to include
- `max_hop`: Maximum distance from starting entities (default: 1)
- `include_paths`: Whether to find paths between entities (default: False)

### 3. `traverse_knowledge_graph`
Traverse the knowledge graph from a single starting node.

**Features:**
- Single-node traversal with configurable depth
- Nested structure showing relationships hierarchically
- Cycle detection to prevent infinite recursion
- Detailed edge information with facts and relationships

**Parameters:**
- `start_node_uuid`: UUID of the starting entity
- `depth`: Traversal depth (0=node only, 1=direct relations, etc.)

## 🎯 Design Philosophy

This extension follows Graphiti-core's design principles:

1. **Entity-Focused**: Only Entity nodes and RELATES_TO edges are included in results
2. **Episodic Exclusion**: Episodic nodes are intentionally excluded as they are for data provenance, not knowledge structure
3. **Embedding Optimization**: All embedding fields are automatically excluded to keep response sizes manageable
4. **MCP Compatibility**: Response sizes are optimized to stay within MCP's 25,000 token limit

## 📊 Performance Optimizations

- **94% response size reduction** through intelligent embedding exclusion
- **Separate queries** for nodes and edges to minimize data transfer
- **Pydantic model_dump** with nested field exclusion for efficient serialization
- **Graphiti core functions** used throughout for consistency and performance

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Pull the pre-built image
docker pull ghcr.io/mkxultra/graphiti-mcp-server-extended:latest

# Run with docker-compose
docker compose up

# Or run standalone
docker run -p 8000:8000 \
  -e NEO4J_URI=bolt://your-neo4j:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=your-password \
  -e OPENAI_API_KEY=your-api-key \
  ghcr.io/mkxultra/graphiti-mcp-server-extended:latest
```

### Option 2: Local Installation

Prerequisites:
1. Python 3.10 or higher
2. Neo4j database (version 5.26+)
3. OpenAI API key for LLM operations

```bash
# Clone this repository
git clone https://github.com/mkXultra/graphiti-mcp-server.git
cd graphiti-mcp-server

# Install dependencies with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Configuration

Create a `.env` file:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_api_key
MODEL_NAME=gpt-4-mini
```

### Running the Server

```bash
# Run with uv
uv run graphiti_mcp_server.py

# Or with Docker
docker compose up
```

## 🔧 MCP Client Configuration

### For Claude Desktop (stdio transport)

```json
{
  "mcpServers": {
    "graphiti-extended": {
      "transport": "stdio",
      "command": "/path/to/uv",
      "args": [
        "run",
        "--directory",
        "/path/to/graphiti-mcp-server",
        "graphiti_mcp_server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "OPENAI_API_KEY": "sk-XXXXXXXX"
      }
    }
  }
}
```

### For Cursor (SSE transport)

```json
{
  "mcpServers": {
    "graphiti-extended": {
      "transport": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

## 📚 Available Tools

### Original Graphiti Tools
- `add_memory`: Add episodes to the knowledge graph
- `search_memory_nodes`: Search for entity nodes
- `search_memory_facts`: Search for relationships
- `delete_entity_edge`: Remove edges
- `delete_episode`: Remove episodes
- `get_entity_edge`: Get edge details
- `get_episodes`: Retrieve recent episodes
- `clear_graph`: Reset the graph

### 🆕 Extended Tools (This Fork)
- `find_paths_between_entities`: Discover paths between entities
- `build_subgraph`: Extract subgraphs around entities
- `traverse_knowledge_graph`: Single-node graph traversal

## 💡 Usage Examples

### Finding Paths Between Entities
```python
# Find how two people are connected
result = await find_paths_between_entities(
    from_uuid="person1_uuid",
    to_uuid="person2_uuid",
    max_depth=3,
    max_paths=5
)
# Returns multiple paths showing relationships
```

### Building a Subgraph
```python
# Extract a local graph around specific entities
result = await build_subgraph(
    entity_uuids=["entity1_uuid", "entity2_uuid"],
    max_hop=2,  # Include entities up to 2 hops away
    include_paths=True  # Find paths between included entities
)
# Returns complete subgraph with nodes, edges, and adjacency list
```

### Traversing from a Single Node
```python
# Traverse the graph from a single starting point
result = await traverse_knowledge_graph(
    start_node_uuid="entity_uuid",
    depth=2  # Traverse 2 levels deep
)
# Returns nested structure showing relationships hierarchically
```

## 🏗️ Architecture

```
graphiti-mcp-server/
├── graphiti_mcp_server.py    # Main MCP server
├── src/
│   └── tools/
│       ├── __init__.py
│       └── graph_functions.py  # New graph manipulation tools
├── tests/
│   └── test_graph_functions.py # Comprehensive test suite
└── docker-compose.yml          # Docker setup with Neo4j
```

## 🧪 Testing

### Prerequisites
The test suite requires a separate Neo4j test database to avoid affecting your main data.

### Running Tests

```bash
# 1. Start the test database
docker-compose -f docker-compose.test.yml up -d

# 2. Wait for the database to be ready
sleep 5

# 3. Run all tests (using uv for dependency management)
OPENAI_API_KEY=dummy uv run python -m pytest src/tests/ -v

# Or run specific test class
OPENAI_API_KEY=dummy uv run python -m pytest src/tests/test_graph_functions.py::TestFindPathsBetweenEntities -v

# Or run specific test method
OPENAI_API_KEY=dummy uv run python -m pytest src/tests/test_graph_functions.py::TestFindPathsBetweenEntities::test_find_direct_path -v

# Alternative: without uv (requires manual dependency installation)
# OPENAI_API_KEY=dummy python3 -m pytest src/tests/ -v

# 4. Stop the test database when done
docker-compose -f docker-compose.test.yml down
```

### Test Database Configuration
- Port: 7688 (Bolt)
- Port: 7475 (HTTP)
- Username: neo4j
- Password: testpassword

**Note:** The `OPENAI_API_KEY` can be any dummy value as it's required by Graphiti core but not actually used in tests.

## 🤝 Contributing

Contributions are welcome! This fork maintains compatibility with the upstream Graphiti project while adding extended functionality.

### Development Guidelines
1. Follow Graphiti-core design patterns
2. Exclude Episodic nodes from knowledge operations
3. Optimize for MCP token limits
4. Use Graphiti core functions where possible
5. Add comprehensive tests for new features

## 📄 License

This project maintains the same license as the parent Graphiti project.

## 🙏 Acknowledgments

- Original [Graphiti](https://github.com/getzep/graphiti) project by Zep
- Built on the Graphiti-core framework
- MCP protocol by Anthropic

## 📊 Technical Details

### Response Size Optimization
- Embeddings excluded using Pydantic's nested field exclusion
- Response sizes reduced from ~93,596 to ~5,861 tokens (94% reduction)
- MCP-compatible responses under 25,000 token limit

### Query Optimization
- Separate queries for paths, nodes, and edges
- Entity-only traversal using RELATES_TO edges
- Batch processing for multiple entities

### Tool Comparison
| Tool | Starting Points | Structure | Best For |
|------|----------------|-----------|----------|
| `find_paths_between_entities` | 2 entities | Paths only | Relationship discovery |
| `build_subgraph` | Multiple entities | Full graph | Neighborhood exploration |
| `traverse_knowledge_graph` | Single entity | Nested hierarchy | Deep traversal |

### Code Quality
- 110+ comprehensive tests with 100% pass rate
- Type hints throughout
- Consistent error handling
- Extensive logging for debugging