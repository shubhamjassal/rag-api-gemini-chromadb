# Retrieval-Augmented Generation (RAG) API

A Python-based Retrieval-Augmented Generation application that retrieves relevant document passages from a ChromaDB vector database and uses Google's Gemini API to generate context-grounded answers. The application exposes its functionality through a FastAPI REST API.

## Features

- **Document ingestion:** Process a PDF, split its content into chunks, and store the resulting embeddings in ChromaDB.
- **Semantic retrieval:** Embed user questions and retrieve relevant document passages.
- **Context-grounded generation:** Generate answers using retrieved document context.
- **REST API:** Expose the RAG pipeline through FastAPI.
- **Evaluation:** Measure retrieval quality using Hit Rate, Precision, Recall, and Mean Reciprocal Rank (MRR).
- **Performance tracking:** Record pipeline latency and token usage, including input, output, and thinking tokens.
- **Resilience:** Retry selected transient Gemini API errors and handle quota exhaustion without long automatic waits.
- **Automated tests:** Test application behavior with pytest.

## Architecture

```text
PDF Document
    |
    v
Document Ingestion
    |
    v
Text Chunking
    |
    v
Gemini Embeddings
    |
    v
ChromaDB Vector Store

User Question
    |
    v
Query Embedding
    |
    v
Semantic Retrieval
    |
    v
Retrieved Context
    |
    v
Gemini Answer Generation
    |
    v
Answer + Sources + Metrics
```

## Technology Stack

- Python
- Google Gemini API (`google-genai`)
- ChromaDB
- FastAPI
- Pydantic
- Uvicorn
- pytest

## Project Structure

```text
RAG_Project/
├── config.py
├── ingest.py
├── rag_engine.py
├── main.py
├── evaluation.py
├── evaluation_dataset.py
├── test_main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.mdREADME.md
```

The repository may also include experimental scripts documenting retrieval, chunking, evaluation, and other RAG improvements.

## Prerequisites

- Python 3.11 or newer recommended; the project has been tested in a Python 3.14 environment.
- A Gemini API key.
- Git, if cloning the repository.

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG_Project
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a local `.env` file based on the example:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your own Gemini API key:

```text
GEMINI_API_KEY=your_actual_api_key
```

Never commit or share your `.env` file.

### 5. Ingest the document

Ensure the PDF is available at the path expected by `ingest.py`, then run:

```powershell
python ingest.py
```

The ingestion process creates or updates the local ChromaDB collection. Follow the script's output to confirm ingestion completed successfully.

### 6. Start the API

```powershell
python -m uvicorn main:app --reload
```

The API should be available at:

- Root endpoint: `http://127.0.0.1:8000/`
- Health endpoint: `http://127.0.0.1:8000/health`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

## API Usage

The application provides a `POST /ask` endpoint for questions.

Example request:

```json
{
  "question": "What is RAG?"
}
```

Use the interactive documentation at `/docs` to inspect the exact request and response schema.

## Evaluation

The project includes retrieval benchmarks using a five-question evaluation dataset.

Previously measured results for the tested `TOP_K` settings:

| TOP_K | Hit Rate | Precision | Recall | MRR |
|---:|---:|---:|---:|---:|
| 1 | 0.400 | 0.400 | 0.400 | 0.400 |
| 3 | 1.000 | 0.500 | 1.000 | 0.667 |
| 5 | 1.000 | 0.417 | 1.000 | 0.667 |
| 10 | 1.000 | 0.213 | 1.000 | 0.667 |

For this small dataset, `TOP_K=3` provided full recall while returning fewer results than larger settings. These figures are preliminary and should not be interpreted as production-level performance.

## Running Tests

```powershell
python -m pytest -q
```

The existing test suite previously reported **7 passed**. Results may vary with code and dependency changes.

## Limitations

- Evaluation uses a small dataset and is not sufficient to establish general performance.
- Answer generation requires Gemini API access and is subject to provider quotas and rate limits.
- The vector database is stored locally and must be populated through ingestion.
- The application is a learning and portfolio project; production deployment would require additional security, monitoring, deployment configuration, and broader evaluation.

## Future Improvements

- Expand the evaluation dataset and ground-truth annotations.
- Add retrieval and answer-quality regression tests.
- Compare embedding models, chunking strategies, and reranking.
- Add authentication, structured logging, and deployment configuration.
- Track cost, latency, and quality across repeatable experiments.

## License

Add a license before distributing the project publicly. If you are unsure which license to use, review the implications before selecting one.
