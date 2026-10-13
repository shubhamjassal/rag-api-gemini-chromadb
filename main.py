import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine import (
    rag_answer,
    health_check,
    RAGError
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API",
    version="1.1.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to ask the RAG system"
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class Source(BaseModel):

    source: str | None = None
    page: int | None = None
    chunk_id: int | None = None


class TimingResponse(BaseModel):

    embedding_seconds: float
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float

class UsageResponse(BaseModel):

    input_tokens: int
    output_tokens: int
    total_tokens: int

class AnswerResponse(BaseModel):

    answer: str
    sources: list[Source]
    latency_seconds: float
    timings: TimingResponse
    usage: UsageResponse

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "RAG API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    status = health_check()

    if status["chroma"]["status"] != "healthy":
        raise HTTPException(
            status_code=503,
            detail=status
        )

    return status


# ============================================================
# ASK
# ============================================================

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):

    start_time = time.perf_counter()

    question = request.question.strip()

    logger.info("Received a question request")


    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not question:

        logger.warning(
            "Empty question received"
        )

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    # --------------------------------------------------------
    # RUN RAG
    # --------------------------------------------------------

    try:

        result = rag_answer(question)


    # --------------------------------------------------------
    # KNOWN RAG ERROR
    # --------------------------------------------------------

    except RAGError:

        logger.exception(
            "RAG error"
        )

        raise HTTPException(
            status_code=503,
            detail="RAG service is temporarily unavailable."
        )


    # --------------------------------------------------------
    # UNEXPECTED ERROR
    # --------------------------------------------------------

    except Exception:

        logger.exception(
            "Unexpected server error"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )


    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Request completed in %.3f seconds",
        elapsed
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "latency_seconds": round(elapsed, 3),
        "timings": result["timings"],
        "usage": result["usage"]
    }


