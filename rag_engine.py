import chromadb
from google import genai
import time
from google.genai import types
import random
from google.genai import errors

from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PATH,
    TOP_K
)


# ============================================================
# RAG ERROR
# ============================================================

class RAGError(Exception):
    """Base exception for RAG errors."""
    pass


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)



def generate_with_retry(prompt: str, max_attempts: int = 3):
    """Retry transient Gemini errors without long quota waits."""

    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )

        except (errors.ServerError, errors.ClientError) as e:
            status_code = getattr(e, "code", None)

            if status_code not in (429, 500, 502, 503, 504):
                raise

            details = getattr(e, "details", None)
            error_data = (
                details.get("error", details)
                if isinstance(details, dict)
                else {}
            )

            message = error_data.get("message", "")
            quota_exhausted = (
                "RESOURCE_EXHAUSTED" in str(error_data.get("status", ""))
                or "Quota exceeded" in message
            )

            # A quota failure may require hours, not seconds.
            # Stop immediately instead of sleeping.
            if status_code == 429 and quota_exhausted:
                raise RuntimeError(
                    "Gemini quota exhausted. "
                    "Check your quota and billing, then retry later. "
                    f"Provider message: {message}"
                ) from e

            if attempt == max_attempts - 1:
                raise

            # Short bounded retries for transient errors.
            delay = min(2 ** (attempt + 1), 8)

            print(
                f"Gemini returned HTTP {status_code}. "
                f"Waiting {delay}s before retry "
                f"{attempt + 2}/{max_attempts}."
            )

            time.sleep(delay)




# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name="documents"
)


# ============================================================
# RAG FUNCTION
# ============================================================


def needs_decomposition(question: str) -> bool:
    concepts = [
        "embedding",
        "chromadb",
        "vector database",
        "semantic search",
        "retrieval",
        "generation",
        "chunking",
        "chunk overlap",
        "hallucination",
        "rag",
    ]

    question_lower = question.lower()

    matches = sum(
        concept in question_lower
        for concept in concepts
    )

    return matches >= 2



def retrieve_documents(query: str, n_results: int = TOP_K):
    """Embed a query and retrieve matching documents."""

    # Measure embedding time
    embedding_start = time.perf_counter()

    embedding_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    query_embedding = embedding_response.embeddings[0].values
    embedding_seconds = time.perf_counter() - embedding_start

    # Measure ChromaDB search time
    retrieval_start = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    retrieval_seconds = time.perf_counter() - retrieval_start

    return results, embedding_seconds, retrieval_seconds

    
    


def rag_answer(question: str):

    # --------------------------------------------------------
    # START TOTAL TIMER
    # --------------------------------------------------------

    total_start = time.perf_counter()


    # --------------------------------------------------------
    # 1. Create query embedding
    # --------------------------------------------------------


    # --------------------------------------------------------
    # 2. Search ChromaDB
    # --------------------------------------------------------

    try:
        embedding_seconds = 0.0
        retrieval_seconds = 0.0

        retrieved = {}

        def add_results(search_results):
            for document, metadata in zip(
                search_results["documents"][0],
                search_results["metadatas"][0],
            ):
                key = (
                    metadata.get("source"),
                    metadata.get("page"),
                    metadata.get("chunk_id"),
                )

                retrieved[key] = {
                    "document": document,
                    "metadata": metadata,
                }

        # First search: the original question
        results, embed_time, search_time = retrieve_documents(
            question, TOP_K
        )

        embedding_seconds += embed_time
        retrieval_seconds += search_time
        add_results(results)

        # Optional second search for multi-concept questions
        if needs_decomposition(question):
            targeted_queries = [
    "Explain how text embeddings represent meaning and how semantic search compares a query with stored document vectors.",
    "Explain how ChromaDB stores document chunks, embeddings, and metadata, and how it retrieves relevant chunks in a RAG system.",
]

            for targeted_query in targeted_queries:
                extra_results, embed_time, search_time = (
                    retrieve_documents(targeted_query, 3)
                )

                embedding_seconds += embed_time
                retrieval_seconds += search_time
                add_results(extra_results)

        documents = [
            item["document"] for item in retrieved.values()
        ]

        metadatas = [
            item["metadata"] for item in retrieved.values()
        ]

    except Exception as e:
        raise RAGError("Document retrieval failed.") from e



    # --------------------------------------------------------
    # 4. Build context
    # --------------------------------------------------------

    context_parts = []

    for i, document in enumerate(documents):

        context_parts.append(
            f"""
SOURCE {i + 1}

{document}
"""
        )

    context = "\n".join(context_parts)


    # --------------------------------------------------------
    # 5. Build prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say:

"I don't have enough information in the provided document."

Context:

{context}

Question:

{question}
"""


    # --------------------------------------------------------
    # 6. Generate answer
    # --------------------------------------------------------

    try:

        generation_start = time.perf_counter()

        response = generate_with_retry(prompt)
        

        usage = {
    "input_tokens": response.usage_metadata.prompt_token_count or 0,
    "output_tokens": response.usage_metadata.candidates_token_count or 0,
    "thinking_tokens": response.usage_metadata.thoughts_token_count or 0,
    "total_tokens": response.usage_metadata.total_token_count or 0
}

        generation_seconds = (
        time.perf_counter() - generation_start
)

    except Exception as e:

        raise RAGError(
            "Answer generation failed."
        ) from e

    answer = (response.text or "").strip()
    if not answer:
        raise RAGError("The model returned an empty answer.")
    # --------------------------------------------------------
    # 7. Prepare sources
    # --------------------------------------------------------

    sources = []

    for metadata in metadatas:

        sources.append({
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id")
        })


    # --------------------------------------------------------
    # 8. Calculate total time
    # --------------------------------------------------------

    total_seconds = (
        time.perf_counter() - total_start
    )


    # --------------------------------------------------------
    # 9. Return result
    # --------------------------------------------------------

    return {
        "answer": answer,

        "context": context,

        "sources": sources,

        "timings": {
            "embedding_seconds": round(embedding_seconds, 3),

            "retrieval_seconds": round(retrieval_seconds, 3),

            "generation_seconds": round(generation_seconds, 3),

            "total_seconds": round(total_seconds, 3) } ,
        "usage": usage  
 }


# ============================================================
# CHROMADB HEALTH CHECK
# ============================================================

def check_chroma():

    try:

        count = collection.count()

        return {
            "status": "healthy",
            "chunks": count
        }

    except Exception as e:

        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ============================================================
# RAG HEALTH CHECK
# ============================================================

def health_check():

    chroma_status = check_chroma()

    return {
        "chroma": chroma_status
    }