"""Session storage and token management for cursor-based pagination."""

import time
import json
import hashlib
import hmac
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta


# Exceptions
class CursorExpired(Exception):
    """Raised when a cursor token has expired."""
    pass


class InvalidCursor(Exception):
    """Raised when a cursor token is invalid or malformed."""
    pass


class SessionNotFound(Exception):
    """Raised when a session cannot be found."""
    pass


class QueryMismatch(Exception):
    """Raised when query parameters don't match the original."""
    pass


@dataclass
class Frame:
    """Represents a node being processed in BFS traversal."""
    node_uuid: str
    depth_remaining: int
    next_edge_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Frame":
        """Create Frame from dictionary."""
        return cls(
            node_uuid=data["node_uuid"],
            depth_remaining=data["depth_remaining"],
            next_edge_index=data["next_edge_index"]
        )


@dataclass
class TraverseSession:
    """Session state for graph traversal."""
    # Immutable query parameters
    root_uuid: str
    max_depth: int
    strategy: str = "bfs"
    edge_ordering: str = "uuid"
    query_hash: str = ""
    snapshot_as_of: Optional[str] = None
    
    # Mutable traversal state
    frontier: List[Frame] = field(default_factory=list)
    visited: List[str] = field(default_factory=list)
    yielded_edges: int = 0
    
    # Session metadata
    started_at: float = field(default_factory=time.time)
    expires_at: float = 0
    schema_version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "root_uuid": self.root_uuid,
            "max_depth": self.max_depth,
            "strategy": self.strategy,
            "edge_ordering": self.edge_ordering,
            "query_hash": self.query_hash,
            "snapshot_as_of": self.snapshot_as_of,
            "frontier": [f.to_dict() for f in self.frontier],
            "visited": self.visited,
            "yielded_edges": self.yielded_edges,
            "started_at": self.started_at,
            "expires_at": self.expires_at,
            "schema_version": self.schema_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraverseSession":
        """Create TraverseSession from dictionary."""
        frontier = [Frame.from_dict(f) for f in data.get("frontier", [])]
        return cls(
            root_uuid=data["root_uuid"],
            max_depth=data["max_depth"],
            strategy=data.get("strategy", "bfs"),
            edge_ordering=data.get("edge_ordering", "uuid"),
            query_hash=data.get("query_hash", ""),
            snapshot_as_of=data.get("snapshot_as_of"),
            frontier=frontier,
            visited=data.get("visited", []),
            yielded_edges=data.get("yielded_edges", 0),
            started_at=data.get("started_at", time.time()),
            expires_at=data.get("expires_at", 0),
            schema_version=data.get("schema_version", 1),
        )


class SessionStore:
    """In-memory session storage with token management."""
    
    # Secret key for token signing (in production, load from environment)
    SECRET_KEY = b"graphiti-mcp-secret-key-change-in-production"
    
    def __init__(self):
        """Initialize the session store."""
        self._sessions: Dict[str, TraverseSession] = {}
    
    async def save_session(self, session_id: str, session: TraverseSession) -> None:
        """Save a session to storage."""
        self._sessions[session_id] = session
    
    async def load_session(self, session_id: str) -> Optional[TraverseSession]:
        """Load a session from storage."""
        return self._sessions.get(session_id)
    
    async def delete_session(self, session_id: str) -> None:
        """Delete a session from storage."""
        self._sessions.pop(session_id, None)
    
    async def clear_all(self) -> None:
        """Clear all sessions (for testing)."""
        self._sessions.clear()
    
    async def issue_token(
        self,
        session_id: str,
        query_hash: str,
        ttl_seconds: int = 600
    ) -> Dict[str, Any]:
        """Issue a signed token for a session."""
        now = time.time()
        exp = now + ttl_seconds
        
        # Create payload
        payload = {
            "sid": session_id,
            "qh": query_hash,
            "iat": int(now),
            "exp": int(exp),
        }
        
        # Encode as JSON
        payload_json = json.dumps(payload, separators=(",", ":"))
        payload_bytes = payload_json.encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
        
        # Create signature
        signature = hmac.new(
            self.SECRET_KEY,
            payload_b64.encode("utf-8"),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
        
        # Combine into token
        token = f"{payload_b64}.{signature_b64}"
        
        return {
            "token": token,
            "expires_at": exp,
        }
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a token."""
        try:
            # Split token
            parts = token.split(".")
            if len(parts) != 2:
                raise InvalidCursor("Malformed token")
            
            payload_b64, signature_b64 = parts
            
            # Verify signature
            expected_signature = hmac.new(
                self.SECRET_KEY,
                payload_b64.encode("utf-8"),
                hashlib.sha256
            ).digest()
            
            # Decode provided signature (add padding if needed)
            signature_b64_padded = signature_b64 + "=" * (4 - len(signature_b64) % 4)
            provided_signature = base64.urlsafe_b64decode(signature_b64_padded)
            
            if not hmac.compare_digest(expected_signature, provided_signature):
                raise InvalidCursor("Invalid signature")
            
            # Decode payload (add padding if needed)
            payload_b64_padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64_padded)
            payload = json.loads(payload_bytes.decode("utf-8"))
            
            # Check expiration
            if "exp" in payload and payload["exp"] < time.time():
                raise CursorExpired("Token has expired")
            
            return payload
            
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            raise InvalidCursor(f"Failed to decode token: {e}")
        except CursorExpired:
            raise
        except Exception as e:
            raise InvalidCursor(f"Token verification failed: {e}")