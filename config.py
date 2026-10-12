import os
from dotenv import load_dotenv


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "gemini-3.6-flash"

EMBEDDING_MODEL = "gemini-embedding-001"

CHROMA_PATH = "./chroma_db"

TOP_K = 3

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )