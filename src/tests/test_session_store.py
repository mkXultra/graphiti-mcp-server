"""Test cases for SessionStore functionality."""

import pytest
import pytest_asyncio
import time
from datetime import datetime, timedelta
from freezegun import freeze_time
from typing import Dict, Any

# Import the module we'll implement
from src.tools.session_store import (
    SessionStore,
    TraverseSession,
    Frame,
    CursorExpired,
    InvalidCursor,
    SessionNotFound,
)


class TestSessionStore:
    """Test cases for SessionStore functionality."""
    
    @pytest_asyncio.fixture
    async def store(self):
        """Create a SessionStore instance for testing."""
        store = SessionStore()
        yield store
        # Cleanup
        await store.clear_all()
    
    @pytest.mark.asyncio
    async def test_session_save_load_roundtrip(self, store):
        """Test saving and loading session preserves data."""
        frame = Frame(node_uuid="N1", depth_remaining=3, next_edge_index=0)
        sess = TraverseSession(
            root_uuid="N1",
            max_depth=3,
            strategy="bfs",
            edge_ordering="uuid",
            query_hash="N1:3",
            frontier=[frame],
            visited=["N1"],
            yielded_edges=0,
            started_at=time.time(),
            expires_at=time.time() + 600,
            schema_version=1
        )
        
        sid = "test-session-123"
        await store.save_session(sid, sess)
        loaded = await store.load_session(sid)
        
        assert loaded is not None
        assert loaded.root_uuid == "N1"
        assert loaded.max_depth == 3
        assert len(loaded.frontier) == 1
        assert loaded.frontier[0].node_uuid == "N1"
        assert loaded.frontier[0].depth_remaining == 3
        assert loaded.frontier[0].next_edge_index == 0
        assert loaded.visited == ["N1"]
    
    @pytest.mark.asyncio
    async def test_session_not_found(self, store):
        """Test loading non-existent session returns None."""
        loaded = await store.load_session("non-existent-id")
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_session_delete(self, store):
        """Test deleting a session."""
        sess = TraverseSession(
            root_uuid="N1",
            max_depth=2,
            strategy="bfs",
            edge_ordering="uuid",
            query_hash="N1:2",
            frontier=[],
            visited=["N1"],
            yielded_edges=0,
            started_at=time.time(),
            expires_at=time.time() + 600,
            schema_version=1
        )
        
        sid = "delete-test"
        await store.save_session(sid, sess)
        
        # Verify it exists
        loaded = await store.load_session(sid)
        assert loaded is not None
        
        # Delete it
        await store.delete_session(sid)
        
        # Verify it's gone
        loaded = await store.load_session(sid)
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_token_issue_and_verify(self, store):
        """Test issuing and verifying a token."""
        sid = "token-test"
        query_hash = "N1:3"
        
        token_info = await store.issue_token(sid, query_hash)
        
        assert "token" in token_info
        assert "expires_at" in token_info
        assert token_info["expires_at"] > time.time()
        
        # Verify the token
        payload = await store.verify_token(token_info["token"])
        
        assert payload["sid"] == sid
        assert payload["qh"] == query_hash
        assert "exp" in payload
        assert "iat" in payload
    
    @pytest.mark.asyncio
    async def test_token_invalid_signature(self, store):
        """Test verifying token with invalid signature raises InvalidCursor."""
        invalid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"
        
        with pytest.raises(InvalidCursor):
            await store.verify_token(invalid_token)
    
    @pytest.mark.asyncio
    async def test_token_malformed(self, store):
        """Test verifying malformed token raises InvalidCursor."""
        malformed_token = "not-a-valid-token"
        
        with pytest.raises(InvalidCursor):
            await store.verify_token(malformed_token)
    
    @pytest.mark.asyncio
    @freeze_time("2024-01-01 12:00:00")
    async def test_token_expiry(self, store):
        """Test token expiration after TTL."""
        sid = "expire-test"
        query_hash = "N1:2"
        
        # Issue token with 10 minute TTL
        token_info = await store.issue_token(sid, query_hash, ttl_seconds=600)
        
        # Token should be valid now
        payload = await store.verify_token(token_info["token"])
        assert payload["sid"] == sid
        
        # Move time forward past TTL (11 minutes)
        with freeze_time("2024-01-01 12:11:00"):
            with pytest.raises(CursorExpired):
                await store.verify_token(token_info["token"])
    
    @pytest.mark.asyncio
    @freeze_time("2024-01-01 12:00:00")
    async def test_sliding_ttl(self, store):
        """Test that reissuing token extends expiration (sliding TTL)."""
        sid = "sliding-test"
        query_hash = "N1:2"
        
        # Issue initial token with 5 minute TTL
        token1 = await store.issue_token(sid, query_hash, ttl_seconds=300)
        
        # Move forward 3 minutes
        with freeze_time("2024-01-01 12:03:00"):
            # Reissue token - should extend TTL
            token2 = await store.issue_token(sid, query_hash, ttl_seconds=300)
            
            # Original token should still be valid (2 minutes left)
            payload1 = await store.verify_token(token1["token"])
            assert payload1["sid"] == sid
            
            # New token should also be valid
            payload2 = await store.verify_token(token2["token"])
            assert payload2["sid"] == sid
        
        # Move forward to 6 minutes from start
        with freeze_time("2024-01-01 12:06:00"):
            # Original token should be expired
            with pytest.raises(CursorExpired):
                await store.verify_token(token1["token"])
            
            # New token should still be valid (2 minutes left from reissue)
            payload2 = await store.verify_token(token2["token"])
            assert payload2["sid"] == sid
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, store):
        """Test that multiple sessions can coexist."""
        # Create multiple sessions
        sessions_data = [
            ("sess1", "N1", 2, ["N1", "N2"]),
            ("sess2", "N3", 3, ["N3"]),
            ("sess3", "N4", 1, ["N4", "N5", "N6"]),
        ]
        
        for sid, root, depth, visited in sessions_data:
            sess = TraverseSession(
                root_uuid=root,
                max_depth=depth,
                strategy="bfs",
                edge_ordering="uuid",
                query_hash=f"{root}:{depth}",
                frontier=[],
                visited=visited,
                yielded_edges=0,
                started_at=time.time(),
                expires_at=time.time() + 600,
                schema_version=1
            )
            await store.save_session(sid, sess)
        
        # Verify all sessions exist and have correct data
        for sid, root, depth, visited in sessions_data:
            loaded = await store.load_session(sid)
            assert loaded is not None
            assert loaded.root_uuid == root
            assert loaded.max_depth == depth
            assert loaded.visited == visited
    
    @pytest.mark.asyncio
    async def test_frame_serialization(self, store):
        """Test that Frame objects with various states serialize correctly."""
        frames = [
            Frame("N1", 3, 0),
            Frame("N2", 2, 5),  # Mid-edge processing
            Frame("N3", 1, 10), # Many edges processed
            Frame("N4", 0, 0),  # Leaf node
        ]
        
        sess = TraverseSession(
            root_uuid="N1",
            max_depth=3,
            strategy="bfs",
            edge_ordering="type_then_uuid",
            query_hash="complex",
            frontier=frames,
            visited=["N1", "N2", "N3"],
            yielded_edges=15,
            started_at=time.time(),
            expires_at=time.time() + 600,
            schema_version=1
        )
        
        sid = "frame-test"
        await store.save_session(sid, sess)
        loaded = await store.load_session(sid)
        
        assert loaded is not None
        assert len(loaded.frontier) == 4
        
        for i, frame in enumerate(loaded.frontier):
            assert frame.node_uuid == frames[i].node_uuid
            assert frame.depth_remaining == frames[i].depth_remaining
            assert frame.next_edge_index == frames[i].next_edge_index