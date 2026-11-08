
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

import backend.rag as rag
from backend.rag import (
    chunk_text,
    search_knowledge_base,
)


# ============================================================
# TEST FIXTURES
# ============================================================

def make_test_index():
    """Create a deterministic FAISS index without API calls."""

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(3)
    index.add(vectors)

    return index


def make_test_chunks():
    return [
        {
            "source": "returns.md",
            "chunk_id": "returns.md:0",
            "text": "Return products within 30 days.",
        },
        {
            "source": "shipping.md",
            "chunk_id": "shipping.md:0",
            "text": "Shipping takes five business days.",
        },
        {
            "source": "warranty.md",
            "chunk_id": "warranty.md:0",
            "text": "Warranty coverage lasts one year.",
        },
    ]


def mock_embedding_client(monkeypatch, vector):
    """Mock OpenAI embeddings to avoid external API calls."""

    embedding_response = SimpleNamespace(
        data=[
            SimpleNamespace(
                embedding=vector,
            )
        ]
    )

    embeddings_api = SimpleNamespace(
        create=lambda **kwargs: embedding_response
    )

    client = SimpleNamespace(
        embeddings=embeddings_api
    )

    monkeypatch.setattr(
        rag,
        "get_client",
        lambda: client,
    )


# ============================================================
# CHUNKING TESTS
# ============================================================

def test_chunk_text_returns_chunks():
    chunks = chunk_text(
        "one two three four five",
        chunk_size=3,
        overlap=1,
    )

    assert chunks == [
        "one two three",
        "three four five",
    ]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "some words",
            chunk_size=3,
            overlap=3,
        )


# ============================================================
# SEMANTIC RETRIEVAL TESTS
# ============================================================

def test_semantic_retrieval_returns_matching_document(monkeypatch):
    chunks = make_test_chunks()
    index = make_test_index()

    mock_embedding_client(
        monkeypatch,
        [1.0, 0.0, 0.0],
    )

    results = search_knowledge_base(
        "return products",
        chunks,
        index,
        top_k=3,
        similarity_threshold=0.5,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "returns.md:0"
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_semantic_retrieval_respects_top_k(monkeypatch):
    chunks = make_test_chunks()
    index = make_test_index()

    mock_embedding_client(
        monkeypatch,
        [1.0, 0.0, 0.0],
    )

    results = search_knowledge_base(
        "find relevant documentation",
        chunks,
        index,
        top_k=2,
        similarity_threshold=0.0,
    )

    assert len(results) == 2


def test_semantic_retrieval_applies_similarity_threshold(monkeypatch):
    chunks = make_test_chunks()
    index = make_test_index()

    mock_embedding_client(
        monkeypatch,
        [1.0, 0.0, 0.0],
    )

    results = search_knowledge_base(
        "return policy",
        chunks,
        index,
        top_k=3,
        similarity_threshold=0.5,
    )

    assert all(
        result["similarity"] >= 0.5
        for result in results
    )

    assert all(
        result["chunk_id"] == "returns.md:0"
        for result in results
    )


def test_semantic_retrieval_sorts_by_similarity(monkeypatch):
    chunks = make_test_chunks()
    index = make_test_index()

    mock_embedding_client(
        monkeypatch,
        [1.0, 1.0, 0.0],
    )

    results = search_knowledge_base(
        "find relevant information",
        chunks,
        index,
        top_k=3,
        similarity_threshold=0.0,
    )

    similarities = [
        result["similarity"]
        for result in results
    ]

    assert similarities == sorted(
        similarities,
        reverse=True,
    )


def test_semantic_retrieval_returns_expected_metadata(monkeypatch):
    chunks = make_test_chunks()
    index = make_test_index()

    mock_embedding_client(
        monkeypatch,
        [1.0, 0.0, 0.0],
    )

    results = search_knowledge_base(
        "return policy",
        chunks,
        index,
        top_k=3,
        similarity_threshold=0.5,
    )

    assert len(results) > 0

    required_fields = {
        "source",
        "chunk_id",
        "text",
        "similarity",
        "semantic_similarity",
        "retrieval_method",
    }

    for result in results:
        assert required_fields.issubset(result.keys())
        assert result["retrieval_method"] == "semantic"


def test_semantic_retrieval_handles_empty_question():
    results = search_knowledge_base(
        "",
        make_test_chunks(),
        make_test_index(),
    )

    assert results == []


def test_semantic_retrieval_handles_empty_chunks():
    results = search_knowledge_base(
        "return policy",
        [],
        make_test_index(),
    )

    assert results == []


def test_semantic_retrieval_handles_invalid_top_k():
    results = search_knowledge_base(
        "return policy",
        make_test_chunks(),
        make_test_index(),
        top_k=0,
    )

    assert results == []
