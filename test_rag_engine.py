
from types import SimpleNamespace

import pytest

import rag_engine


def test_rag_answer_rejects_empty_model_response(monkeypatch):
    def fake_retrieve_documents(query, n_results=None):
        return (
            {
                "documents": [["Test document content"]],
                "metadatas": [[
                    {
                        "source": "document.pdf",
                        "page": 1,
                        "chunk_id": 0,
                    }
                ]],
            },
            0.01,
            0.01,
        )

    fake_response = SimpleNamespace(
        text="   ",
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=0,
            thoughts_token_count=0,
            total_token_count=10,
        ),
    )

    monkeypatch.setattr(
        rag_engine,
        "retrieve_documents",
        fake_retrieve_documents,
    )
    monkeypatch.setattr(
        rag_engine,
        "generate_with_retry",
        lambda prompt: fake_response,
    )
    monkeypatch.setattr(
        rag_engine,
        "needs_decomposition",
        lambda question: False,
    )

    with pytest.raises(
        rag_engine.RAGError,
        match="empty answer",
    ):
        rag_engine.rag_answer("What is RAG?")
