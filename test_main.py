from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "RAG API is running"
    }

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["chroma"]["status"] == "healthy"
    assert response.json()["chroma"]["chunks"] == 15

def test_empty_question_rejected():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_whitespace_only_question_rejected():
    response = client.post(
        "/ask",
        json={"question": "   \t  "}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_question_too_long_rejected():
    response = client.post(
        "/ask",
        json={"question": "a" * 1001}
    )

    assert response.status_code == 422

def test_rag_error_returns_503(monkeypatch):
    from rag_engine import RAGError
    import main

    def fake_rag_answer(question):
        raise RAGError("Simulated failure")

    monkeypatch.setattr(main, "rag_answer", fake_rag_answer)

    response = client.post(
        "/ask",
        json={"question": "What is RAG?"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "RAG service is temporarily unavailable."
    )

def test_unexpected_error_returns_500(monkeypatch):
    import main

    def fake_rag_answer(question):
        raise ValueError("Simulated internal failure")

    monkeypatch.setattr(main, "rag_answer", fake_rag_answer)

    response = client.post(
        "/ask",
        json={"question": "What is RAG?"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."

def test_successful_ask_response(monkeypatch):
    import main

    def fake_rag_answer(question):
        return {
            "answer": "Embeddings represent text as vectors.",
            "sources": [
                {"source": "document.pdf", "page": 2, "chunk_id": 4}
            ],
            "timings": {
                "embedding_seconds": 0.1,
                "retrieval_seconds": 0.2,
                "generation_seconds": 0.3,
                "total_seconds": 0.6,
            },
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        }

    monkeypatch.setattr(main, "rag_answer", fake_rag_answer)

    response = client.post(
        "/ask",
        json={"question": "What are embeddings?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Embeddings represent text as vectors."
    assert data["sources"][0]["page"] == 2
    assert data["usage"]["total_tokens"] == 120
    assert data["latency_seconds"] >= 0

def test_unhealthy_health_endpoint_returns_503(monkeypatch):
    import main

    def fake_health_check():
        return {
            "chroma": {
                "status": "unhealthy",
                "error": "Simulated database failure",
            }
        }

    monkeypatch.setattr(main, "health_check", fake_health_check)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"]["chroma"]["status"] == "unhealthy"