
from pathlib import Path

import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader

from config import GEMINI_API_KEY, EMBEDDING_MODEL, CHROMA_PATH

client = genai.Client(api_key=GEMINI_API_KEY)


def chunk_text(text, chunk_size=1000, overlap=150):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def create_document_embedding(text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )
    return response.embeddings[0].values


def main():
    pdf_path = Path(__file__).resolve().parent / "document.pdf"
    reader = PdfReader(str(pdf_path))

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append({"page": page_number, "text": text})

    print(f"PDF loaded: {len(pages)} pages")

    all_chunks = []
    all_metadata = []
    all_ids = []

    for page in pages:
        for chunk in chunk_text(page["text"]):
            chunk_id = len(all_chunks)
            all_chunks.append(chunk)
            all_metadata.append({
                "source": pdf_path.name,
                "page": page["page"],
                "chunk_id": chunk_id,
            })
            all_ids.append(f"chunk_{chunk_id}")

    print(f"Total chunks: {len(all_chunks)}")
    print("Creating embeddings...")

    embeddings = []
    for i, chunk in enumerate(all_chunks, start=1):
        print(f"Embedding {i}/{len(all_chunks)}")
        embeddings.append(create_document_embedding(chunk))

    db = chromadb.PersistentClient(path=CHROMA_PATH)

    # Build and populate a staging collection before replacing production.
    
    # Build and populate a staging collection.
    staging_name = "documents_staging"

    try:
        db.delete_collection(staging_name)
    except Exception:
        pass

    staging = db.create_collection(name=staging_name)
    staging.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )

    # Verify the staged data before changing production.
    if staging.count() != len(all_chunks):
        raise RuntimeError("Staging collection verification failed")

    staged_check = staging.get(include=["documents", "metadatas"])
    if len(staged_check["documents"]) != len(all_chunks):
        raise RuntimeError("Staging document verification failed")

    # Stop if production is unexpectedly missing.
    try:
        production = db.get_collection(name="documents")
    except Exception as e:
        db.delete_collection(staging_name)
        raise RuntimeError(
            "Production collection is missing; refusing automatic replacement."
        ) from e

    # Update production only after staging verification.
    production.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )

    if production.count() != len(all_chunks):
        raise RuntimeError(
            "Production count verification failed; inspect the database."
        )

    db.delete_collection(staging_name)

    print("Ingestion complete.")
    print("Documents in database:", production.count())



if __name__ == "__main__":
    main()
