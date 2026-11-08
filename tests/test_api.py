
import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.main import app
import backend.main as main


client = TestClient(app)


# ============================================================
# EXISTING API TESTS
# ============================================================

def test_health_check(monkeypatch):
    """API health endpoint returns operational status."""

    monkeypatch.setattr(
        main,
        "get_knowledge_base",
        lambda: ([{"source": "shipping.md"}] * 6, None)
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["knowledge_chunks"] == 6


def test_health_check_failure(monkeypatch):
    """API returns 503 when the knowledge base is unavailable."""

    def failing_knowledge_base():
        raise RuntimeError("Knowledge base unavailable")

    monkeypatch.setattr(
        main,
        "get_knowledge_base",
        failing_knowledge_base
    )

    response = client.get("/health")

    assert response.status_code == 503


def test_ask_success(monkeypatch):
    """Valid question returns the expected response structure."""

    mock_answer = {
        "answer": "Standard shipping takes 3 to 5 business days.",
        "sources": [
            {
                "source": "shipping.md",
                "chunk_id": "shipping.md:0",
                "text": "Standard shipping takes 3 to 5 business days.",
                "similarity": 0.85
            }
        ],
        "latency_ms": 1200.0,
        "model": "gpt-5-mini",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
    }

    monkeypatch.setattr(
        main,
        "answer_question",
        lambda question: mock_answer
    )

    response = client.post(
        "/ask",
        json={"question": "How long does shipping take?"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == mock_answer["answer"]
    assert response.json()["model"] == "gpt-5-mini"
    assert len(response.json()["sources"]) == 1


def test_ask_empty_question():
    """Empty questions are rejected."""

    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 422


def test_ask_missing_question():
    """Requests without a question are rejected."""

    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_question_too_long():
    """Questions exceeding 2000 characters are rejected."""

    response = client.post(
        "/ask",
        json={"question": "A" * 2001}
    )

    assert response.status_code == 422


def test_ask_value_error(monkeypatch):
    """ValueError from the service becomes HTTP 422."""

    def failing_answer(question):
        raise ValueError("Invalid question")

    monkeypatch.setattr(
        main,
        "answer_question",
        failing_answer
    )

    response = client.post(
        "/ask",
        json={"question": "Test question"}
    )

    assert response.status_code == 422


def test_ask_unexpected_error(monkeypatch):
    """Unexpected service errors become HTTP 502."""

    def failing_answer(question):
        raise RuntimeError("Internal provider error")

    monkeypatch.setattr(
        main,
        "answer_question",
        failing_answer
    )

    response = client.post(
        "/ask",
        json={"question": "Test question"}
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The AI service could not complete the request."
    )


# ============================================================
# NEW REQUEST TRACING TESTS
# ============================================================

def test_request_id_header_exists(monkeypatch):
    """Every request receives an X-Request-ID header."""

    monkeypatch.setattr(
        main,
        "get_knowledge_base",
        lambda: ([], None)
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert response.headers["x-request-id"]


def test_request_id_is_valid_uuid(monkeypatch):
    """Request IDs use valid UUID formatting."""

    monkeypatch.setattr(
        main,
        "get_knowledge_base",
        lambda: ([], None)
    )

    response = client.get("/health")

    request_id = response.headers["x-request-id"]

    parsed_uuid = uuid.UUID(request_id)

    assert str(parsed_uuid) == request_id
    assert parsed_uuid.version == 4


def test_request_ids_are_unique(monkeypatch):
    """Separate HTTP requests receive different IDs."""

    monkeypatch.setattr(
        main,
        "get_knowledge_base",
        lambda: ([], None)
    )

    response_one = client.get("/health")
    response_two = client.get("/health")

    assert response_one.status_code == 200
    assert response_two.status_code == 200

    assert (
        response_one.headers["x-request-id"]
        != response_two.headers["x-request-id"]
    )


def test_request_id_matches_application_logs(monkeypatch, caplog):
    """Application log records contain the response request ID."""

    monkeypatch.setattr(
        main,
        "get_knowledge_base",
        lambda: ([], None)
    )

    with caplog.at_level(logging.INFO):
        response = client.get("/health")

    request_id = response.headers["x-request-id"]

    matching_records = [
        record
        for record in caplog.records
        if record.name == "backend.main"
        and record.getMessage() == "Health check successful"
        and getattr(record, "request_id", None) == request_id
    ]

    assert response.status_code == 200
    assert len(matching_records) == 1


def test_failed_request_has_request_id(monkeypatch):
    """Handled API failures still receive a request ID."""

    def failing_answer(question):
        raise RuntimeError("Simulated provider failure")

    monkeypatch.setattr(
        main,
        "answer_question",
        failing_answer
    )

    response = client.post(
        "/ask",
        json={"question": "Test failure"}
    )

    assert response.status_code == 502
    assert "x-request-id" in response.headers

    request_id = response.headers["x-request-id"]

    assert uuid.UUID(request_id).version == 4

    assert response.json()["detail"] == (
        "The AI service could not complete the request."
    )


def test_failed_request_is_logged(monkeypatch, caplog):
    """Unexpected service failures are recorded in application logs."""

    def failing_answer(question):
        raise RuntimeError("Simulated provider failure")

    monkeypatch.setattr(
        main,
        "answer_question",
        failing_answer
    )

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/ask",
            json={"question": "Test logging"}
        )

    request_id = response.headers["x-request-id"]

    matching_records = [
        record
        for record in caplog.records
        if record.name == "backend.main"
        and record.getMessage() == "API question request failed"
        and getattr(record, "request_id", None) == request_id
    ]

    assert response.status_code == 502
    assert len(matching_records) == 1
    assert matching_records[0].exc_info is not None
