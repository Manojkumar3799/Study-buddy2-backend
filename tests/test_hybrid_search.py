import pytest
from unittest.mock import MagicMock, patch
from app.services.retrieval_service import retrieve_relevant_chunks, RetrievedChunk
from app.core.exceptions import VectorStoreNotFoundError


@pytest.mark.asyncio
async def test_hybrid_search_queries_both():
    """Verify that hybrid search calls RRF and applies threshold correctly."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Mock count = 5 chunks exists for document
    mock_cur.fetchone.return_value = [5]
    
    # Mock return rows: (chunk_id, text, start_page, end_page, fused_score, raw_similarity, matched_vector, matched_fts)
    mock_cur.fetchall.return_value = [
        (10, "Hello chunk 1", 1, 1, 0.95, 0.95, 1, 1),
        (11, "Hello chunk 2", 2, 2, 0.45, 0.45, 1, 0),
        (12, "Filtered chunk", 3, 3, 0.20, 0.10, 0, 1),  # Should be filtered by similarity_threshold (0.35)
    ]

    with (
        patch("app.services.retrieval_service.get_db_pool", return_value=mock_pool),
        patch("app.services.retrieval_service.embed_question", return_value=[0.1, 0.2, 0.3])
    ):
        results = await retrieve_relevant_chunks(
            document_id="doc-123",
            user_id="test-user-uuid",
            question="What is hello?",
            top_k=5,
            similarity_threshold=0.35,
            search_mode="hybrid"
        )

    # Chunks passing threshold = 2
    assert len(results) == 2
    assert results[0].chunk_id == 10
    assert results[0].score == 0.95
    assert results[1].chunk_id == 11
    assert results[1].score == 0.45

    # Check query executions — user_id is included in the COUNT query for per-user isolation
    mock_cur.execute.assert_any_call(
        "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s AND user_id = %s",
        ("doc-123", "test-user-uuid"),
    )

    # Verify RRF query was called
    called_queries = [call.args[0] for call in mock_cur.execute.call_args_list if call.args]
    assert any("WITH vector_search AS" in q and "fts_search AS" in q for q in called_queries)


@pytest.mark.asyncio
async def test_hybrid_search_vector_only():
    """Verify that search_mode='vector' executes pure pgvector query."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchone.return_value = [5]
    mock_cur.fetchall.return_value = [
        (10, "Hello vector 1", 1, 1, 0.85),
    ]

    with (
        patch("app.services.retrieval_service.get_db_pool", return_value=mock_pool),
        patch("app.services.retrieval_service.embed_question", return_value=[0.1, 0.2, 0.3])
    ):
        results = await retrieve_relevant_chunks(
            document_id="doc-123",
            user_id="test-user-uuid",
            question="What is hello?",
            top_k=5,
            similarity_threshold=0.35,
            search_mode="vector"
        )

    assert len(results) == 1
    assert results[0].chunk_id == 10
    assert results[0].score == 0.85

    # Verify pure vector query was called (no WITH vector_search)
    called_queries = [call.args[0] for call in mock_cur.execute.call_args_list if call.args]
    assert any("1 - (embedding <=> %s::vector)" in q and "WITH" not in q for q in called_queries)


@pytest.mark.asyncio
async def test_hybrid_search_document_not_found():
    """Verify that retrieval raises VectorStoreNotFoundError if document chunk count is 0."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cur = MagicMock()

    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchone.return_value = [0]  # No chunks found for this document

    with (
        patch("app.services.retrieval_service.get_db_pool", return_value=mock_pool),
        patch("app.services.retrieval_service.embed_question", return_value=[0.1, 0.2, 0.3])
    ):
        with pytest.raises(VectorStoreNotFoundError):
            await retrieve_relevant_chunks(
                document_id="nonexistent-doc",
                user_id="test-user-uuid",
                question="Any question?",
            )
