EVALUATION_DATASET = [

    {
        "question": "What is RAG?",
        "relevant_pages": [3],

        "expected_answer": (
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant information from external documents "
            "and provides that information to an LLM so the model can "
            "generate an answer using the retrieved context."
        ),

        "key_facts": [
            "retrieval",
            "external information",
            "LLM",
            "generated answer"
        ]
    },


    {
        "question": "What is ChromaDB?",
        "relevant_pages": [2],

        "expected_answer": (
            "ChromaDB is a vector database commonly used in AI and RAG "
            "applications. It stores documents, embeddings, IDs, and metadata "
            "and allows relevant information to be retrieved using vector search."
        ),

        "key_facts": [
            "vector database",
            "embeddings",
            "documents",
            "metadata",
            "vector search"
        ]
    },


    {
        "question": "Why are embeddings useful for semantic search?",
        "relevant_pages": [2],

        "expected_answer": (
            "Embeddings represent text as numerical vectors. Texts with "
            "similar meanings tend to have similar vectors, allowing semantic "
            "search to retrieve relevant information even when the exact "
            "words in the query do not appear in the document."
        ),

        "key_facts": [
            "numerical vectors",
            "similar meanings",
            "semantic search",
            "relevant information",
            "exact words are not required"
        ]
    },


    {
        "question": "What is chunk overlap?",
        "relevant_pages": [3],

        "expected_answer": (
            "Chunk overlap means that consecutive document chunks share "
            "some of the same text. Overlap helps preserve context that "
            "might otherwise be split across chunk boundaries."
        ),

        "key_facts": [
            "consecutive chunks share text",
            "preserve context",
            "chunk boundaries"
        ]
    },


    {
        "question": "What is a hallucination?",
        "relevant_pages": [4],

        "expected_answer": (
            "A hallucination is information produced by an LLM that is "
            "unsupported, incorrect, or fabricated."
        ),

        "key_facts": [
            "unsupported information",
            "incorrect information",
            "fabricated information"
        ]
    }
]