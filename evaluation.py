
import json

import chromadb
from google import genai

from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PATH,
    TOP_K,
)

from evaluation_dataset import EVALUATION_DATASET
from rag_engine import rag_answer


# ============================================================
# TOP_K EXPERIMENT CONFIGURATION
# ============================================================

TOP_K_VALUES = [1, 3, 5, 10]


# ============================================================
# GEMINI CLIENT
# ============================================================

judge_client = genai.Client(
    api_key=GEMINI_API_KEY
)


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
# RETRIEVAL
# ============================================================

def retrieve_chunks(question, top_k):
    """
    Generate an embedding for the question
    and retrieve top_k chunks from ChromaDB.

    Returns a list of metadata dictionaries.
    """

    embedding_response = judge_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
    )

    query_embedding = embedding_response.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas"],
    )

    metadatas = results["metadatas"][0]

    return metadatas


# ============================================================
# RETRIEVAL EVALUATION METRICS
# ============================================================

def calculate_hit(relevant_pages, retrieved_pages):
    """Return 1 if any relevant page was retrieved; otherwise 0."""

    return int(
        any(
            page in retrieved_pages
            for page in relevant_pages
        )
    )


def calculate_precision(relevant_pages, retrieved_pages):
    """
    Calculate page-level precision.

    Duplicate chunks from the same page count as one page.
    """

    unique_pages = set(retrieved_pages)

    if not unique_pages:
        return 0.0

    relevant_unique_pages = set(relevant_pages)

    relevant_retrieved = len(
        unique_pages & relevant_unique_pages
    )

    return relevant_retrieved / len(unique_pages)


def calculate_recall(relevant_pages, retrieved_pages):
    """
    Calculate page-level recall.

    Measures how many relevant pages were retrieved.
    """

    if not relevant_pages:
        return 0.0

    retrieved_unique_pages = set(retrieved_pages)
    relevant_unique_pages = set(relevant_pages)

    relevant_retrieved = len(
        relevant_unique_pages & retrieved_unique_pages
    )

    return relevant_retrieved / len(relevant_unique_pages)


def calculate_reciprocal_rank(
    relevant_pages,
    retrieved_pages,
):
    """
    Return the reciprocal rank of the first relevant result.

    Note: ranking is based on retrieved chunks, so repeated
    pages can affect the rank.
    """

    for rank, page in enumerate(
        retrieved_pages,
        start=1,
    ):
        if page in relevant_pages:
            return 1.0 / rank

    return 0.0


# ============================================================
# TOP_K EVALUATION
# ============================================================

def evaluate_retrieval_for_k(top_k):
    """
    Evaluate retrieval for a specific top_k value.

    Failed questions are reported separately and excluded
    from the averages of successfully evaluated questions.
    """

    total_questions = len(EVALUATION_DATASET)

    successful_questions = 0
    failed_questions = 0

    total_hits = 0
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVALUATION - TOP_K={top_k}")
    print("=" * 60)

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1,
    ):
        question = item["question"]
        relevant_pages = item["relevant_pages"]

        print("\n" + "-" * 60)
        print(f"QUESTION {index}/{total_questions}")
        print("-" * 60)

        print("Question:", question)
        print("Expected pages:", relevant_pages)

        try:
            metadatas = retrieve_chunks(
                question,
                top_k,
            )

            retrieved_pages = [
                metadata.get("page")
                for metadata in metadatas
            ]

            print("Retrieved pages:", retrieved_pages)

            hit = calculate_hit(
                relevant_pages,
                retrieved_pages,
            )

            precision = calculate_precision(
                relevant_pages,
                retrieved_pages,
            )

            recall = calculate_recall(
                relevant_pages,
                retrieved_pages,
            )

            reciprocal_rank = calculate_reciprocal_rank(
                relevant_pages,
                retrieved_pages,
            )

            print("Hit:", hit)
            print("Precision:", round(precision, 3))
            print("Recall:", round(recall, 3))
            print(
                "Reciprocal Rank:",
                round(reciprocal_rank, 3),
            )

            total_hits += hit
            total_precision += precision
            total_recall += recall
            total_mrr += reciprocal_rank

            successful_questions += 1

        except Exception as e:
            failed_questions += 1

            print("\nRETRIEVAL FAILED:")
            print(f"{type(e).__name__}: {e}")

    print("\n" + "-" * 60)
    print("TOP_K RESULTS")
    print("-" * 60)
    print("Total questions:", total_questions)
    print("Successful:", successful_questions)
    print("Failed:", failed_questions)

    if successful_questions == 0:
        print("No questions were evaluated successfully.")

        return {
            "top_k": top_k,
            "total_questions": total_questions,
            "successful_questions": 0,
            "failed_questions": failed_questions,
            "hit": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
        }

    average_hit = total_hits / successful_questions
    average_precision = total_precision / successful_questions
    average_recall = total_recall / successful_questions
    average_mrr = total_mrr / successful_questions

    return {
        "top_k": top_k,
        "total_questions": total_questions,
        "successful_questions": successful_questions,
        "failed_questions": failed_questions,
        "hit": average_hit,
        "precision": average_precision,
        "recall": average_recall,
        "mrr": average_mrr,
    }


# ============================================================
# TOP_K EXPERIMENT
# ============================================================

def run_top_k_experiment():
    """Compare retrieval quality for different top_k values."""

    print("\n")
    print("#" * 60)
    print("TOP_K EXPERIMENT MATRIX")
    print("#" * 60)

    results = []

    for top_k in TOP_K_VALUES:
        result = evaluate_retrieval_for_k(top_k)
        results.append(result)

    print("\n")
    print("#" * 78)
    print("TOP_K COMPARISON")
    print("#" * 78)

    print(
        f"{'TOP_K':<8}"
        f"{'Success':<10}"
        f"{'Failed':<9}"
        f"{'Hit':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
        f"{'MRR':<10}"
    )

    print("-" * 78)

    for result in results:
        print(
            f"{result['top_k']:<8}"
            f"{result['successful_questions']:<10}"
            f"{result['failed_questions']:<9}"
            f"{result['hit']:<10.3f}"
            f"{result['precision']:<12.3f}"
            f"{result['recall']:<10.3f}"
            f"{result['mrr']:<10.3f}"
        )

    return results


# ============================================================
# GENERATION EVALUATION - LLM AS A JUDGE
# ============================================================

def evaluate_generation(
    question,
    context,
    answer,
):
    """
    Evaluate faithfulness, answer relevance, and groundedness.

    Returns validated scores between 0.0 and 1.0.
    """

    judge_prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate the answer using ONLY the provided context.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate these three criteria.

1. Faithfulness:
How well is the answer supported by the provided context?

2. Answer relevance:
How directly does the answer address the question?

3. Groundedness:
How well are the claims in the answer grounded in the context?

Scoring:
0.0 = completely poor
0.5 = partially good
1.0 = completely good

Return ONLY valid JSON.
Do not use Markdown code fences.

Use exactly this format:

{{
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "groundedness": 0.0
}}
"""

    try:
        response = judge_client.models.generate_content(
            model=LLM_MODEL,
            contents=judge_prompt,
        )

        if not response.text:
            raise ValueError(
                "The evaluation model returned an empty response."
            )

        text = response.text.strip()

        # Remove Markdown fences if the model adds them.
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()
        scores = json.loads(text)

        required_fields = [
            "faithfulness",
            "answer_relevance",
            "groundedness",
        ]

        for field in required_fields:
            if field not in scores:
                raise ValueError(
                    f"Missing evaluation field: {field}"
                )

            value = scores[field]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"{field} must be numeric."
                )

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{field} must be between 0 and 1."
                )

        return scores

    except Exception as e:
        print("\nGeneration evaluation failed:")
        print(f"{type(e).__name__}: {e}")
        raise


# ============================================================
# FULL RAG EVALUATION
# ============================================================

def run_evaluation():
    """
    Evaluate the full RAG pipeline.

    Retrieval, latency, and token averages use successfully
    completed RAG calls. Generation-judge averages use only
    successful judge evaluations.
    """

    print("=" * 60)
    print("RAG EVALUATION")
    print("=" * 60)

    total_questions = len(EVALUATION_DATASET)

    successful_questions = 0
    failed_questions = 0

    # Retrieval metrics
    total_hits = 0
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    # Performance metrics
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_thinking_tokens = 0

    # Generation metrics
    total_faithfulness = 0.0
    total_answer_relevance = 0.0
    total_groundedness = 0.0
    generation_evaluations = 0

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1,
    ):
        question = item["question"]
        relevant_pages = item["relevant_pages"]

        print("\n" + "-" * 60)
        print(f"QUESTION {index}/{total_questions}")
        print("-" * 60)
        print("Question:", question)
        print("Expected pages:", relevant_pages)

        try:
            # ------------------------------------------------
            # Run the RAG pipeline
            # ------------------------------------------------

            result = rag_answer(question)

            successful_questions += 1

            # ------------------------------------------------
            # Retrieval metrics
            # ------------------------------------------------

            retrieved_pages = [
                source.get("page")
                for source in result["sources"]
            ]

            print("Retrieved pages:", retrieved_pages)

            hit = calculate_hit(
                relevant_pages,
                retrieved_pages,
            )

            precision = calculate_precision(
                relevant_pages,
                retrieved_pages,
            )

            recall = calculate_recall(
                relevant_pages,
                retrieved_pages,
            )

            reciprocal_rank = calculate_reciprocal_rank(
                relevant_pages,
                retrieved_pages,
            )

            print("Hit:", hit)
            print("Precision:", round(precision, 3))
            print("Recall:", round(recall, 3))
            print(
                "Reciprocal Rank:",
                round(reciprocal_rank, 3),
            )

            total_hits += hit
            total_precision += precision
            total_recall += recall
            total_mrr += reciprocal_rank

            # ------------------------------------------------
            # Performance metrics
            # ------------------------------------------------

            timings = result["timings"]
            usage = result["usage"]

            print("\nTIMINGS")
            print(
                "Embedding:",
                timings["embedding_seconds"],
                "seconds",
            )
            print(
                "Retrieval:",
                timings["retrieval_seconds"],
                "seconds",
            )
            print(
                "Generation:",
                timings["generation_seconds"],
                "seconds",
            )
            print(
                "Total:",
                timings["total_seconds"],
                "seconds",
            )

            print("\nTOKEN USAGE")
            print("Input:", usage["input_tokens"])
            print("Output:", usage["output_tokens"])
            print("Total:", usage["total_tokens"])

            total_latency += timings["total_seconds"]
            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]
            total_tokens += usage["total_tokens"]
            total_thinking_tokens += usage.get("thinking_tokens", 0)

            # ------------------------------------------------
            # Generation evaluation
            # ------------------------------------------------

            try:
                generation_evaluation = evaluate_generation(
                    question,
                    result["context"],
                    result["answer"],
                )

                print("\nGENERATION EVALUATION:")
                print(generation_evaluation)

                total_faithfulness += (
                    generation_evaluation["faithfulness"]
                )
                total_answer_relevance += (
                    generation_evaluation["answer_relevance"]
                )
                total_groundedness += (
                    generation_evaluation["groundedness"]
                )

                generation_evaluations += 1

            except Exception as e:
                print("\nGENERATION EVALUATION FAILED:")
                print(f"{type(e).__name__}: {e}")
                print(
                    "The RAG result remains a successful "
                    "RAG evaluation."
                )

        except Exception as e:
            failed_questions += 1

            print("\nEVALUATION FAILED:")
            print(f"{type(e).__name__}: {e}")
            print(
                "Skipping this question and continuing "
                "with the remaining dataset."
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    print("\nTotal questions:", total_questions)
    print("Successful:", successful_questions)
    print("Failed:", failed_questions)

    # --------------------------------------------------------
    # Performance and retrieval summaries
    # --------------------------------------------------------

    if successful_questions > 0:
        average_latency = (
            total_latency / successful_questions
        )
        average_input_tokens = (
            total_input_tokens / successful_questions
        )
        average_output_tokens = (
            total_output_tokens / successful_questions
        )
        average_total_tokens = (
            total_tokens / successful_questions
        )
        
        average_thinking_tokens = (
            total_thinking_tokens / successful_questions
        )


        print("\nPERFORMANCE")
        print(
            "Average latency:",
            round(average_latency, 3),
            "seconds",
        )
        print(
            "Average input tokens:",
            round(average_input_tokens, 2),
        )
        print(
            "Average output tokens:",
            round(average_output_tokens, 2),
        )
        print(
            "Average thinking tokens:",
            round(average_thinking_tokens, 2)
        )
        print(
            "Average total tokens:",
            round(average_total_tokens, 2),
        )

        average_hit = total_hits / successful_questions
        average_precision = (
            total_precision / successful_questions
        )
        average_recall = (
            total_recall / successful_questions
        )
        average_mrr = total_mrr / successful_questions

        print("\nRETRIEVAL QUALITY")
        print(
            f"Hit@{TOP_K}:",
            round(average_hit, 3),
        )
        print(
            f"Precision@{TOP_K}:",
            round(average_precision, 3),
        )
        print(
            f"Recall@{TOP_K}:",
            round(average_recall, 3),
        )
        print("MRR:", round(average_mrr, 3))

    else:
        print("\nNo successful RAG evaluations.")

    # --------------------------------------------------------
    # Generation summary
    # --------------------------------------------------------

    if generation_evaluations > 0:
        average_faithfulness = (
            total_faithfulness / generation_evaluations
        )
        average_answer_relevance = (
            total_answer_relevance / generation_evaluations
        )
        average_groundedness = (
            total_groundedness / generation_evaluations
        )

        print("\nGENERATION QUALITY")
        print(
            "Generation evaluations:",
            generation_evaluations,
        )
        print(
            "Average Faithfulness:",
            round(average_faithfulness, 3),
        )
        print(
            "Average Answer Relevance:",
            round(average_answer_relevance, 3),
        )
        print(
            "Average Groundedness:",
            round(average_groundedness, 3),
        )

    else:
        print("\nNo successful generation evaluations.")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


# ============================================================
# RUN EVALUATION
# ============================================================

if __name__ == "__main__":
    run_top_k_experiment()
